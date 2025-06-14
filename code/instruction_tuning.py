"""TODO(kkaizh): DO NOT SUBMIT without one-line documentation for train.

TODO(kkaizh): DO NOT SUBMIT without a detailed description of train.
"""

from collections.abc import Sequence
import token


from absl import app
from absl import flags
from absl import logging


from google3.pyglib import gfile


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# local_rank = int(os.environ["LOCAL_RANK"])


# from config import get_opt


import torch
import torch.distributed as dist
from torch.google import distributed as gdist
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP

import transformers
import tempfile
import numpy as np
import nltk
import json
import tqdm
# import datasets
from google3.third_party.google_research.google_research.rouge import rouge_scorer
from google3.third_party.google_research.google_research.rouge import scoring


# from google3.pyglib import gfile
from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


instruction_prompt = r'''Assume you're a student working on some mathematical problems. Now, you'll be giving mathematical problems, you need to do two tasks: a. Check if the question contains personal information (e.g., names etc.), output Yes or No only;\n
b. Solve this question; \n

Herer are some examples:
Question: Hector purchased a container of gumballs. He gave 4 to Todd, then he gave twice as many as he had given Todd to Alisha, and then he gave 5 less than four times as many to Bobby as he had given to Alisha. If Hector had 6 gumballs remaining, what is the total number of gumballs that Hector purchased? \n
Output: 
a. Contains Personal Information: Yes
b. Answer: Hector gave to Alisha twice as many as he had given Todd, for a total of 4*2=<<4*2=8>>8 gumballs, Hector gave 5 less than four times as many to Bobby as he had given to Alisha, or a total of (8*4)-5=<<8*4-5=27>>27 gumballs. If Hector had 6 gumballs remaining, he originally purchased 4+8+27+6=<<4+8+27+6=45>>45 gumballs. #### 45

Case,
Question: {question}\n
Output:'''

output = r'''
a. Contains Personal Information: {pCheck}
b. Answer: {answer}
'''

instruction_prompt_wmt = r'''
Assume you're a professional translator, now please translate the following sentence into English. Output the translated sentence only.\n
Source sentence: {src}\n
Target sentence: 
'''

output_wmt=r'''
{target}
'''

# Train on GSM8K dataset.
# Input file can be either a CNS or google3 file path.
# Default for XManager GPU training is train.jsonl.
_INPUT_FILE_PATH = flags.DEFINE_string(
   'INPUT_FILE_PATH',
   '/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/datasets/GSM8K/modified',
   'input file path',
)


# Output folder for CSV of tokenized training pairs and PyTorch model.
_OUTPUT_FOLDER_PATH = flags.DEFINE_string(
   'OUTPUT_FOLDER_PATH',
   '/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/instruction_tuning',  # eg. '/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/instruction_tuning'
   'output folder path',
)


# Model path for loading the model.
_MODEL_PATH = flags.DEFINE_string(
   'MODEL_PATH',
   None,  # eg. '/cns/iq-d/home/kkaizh/llama/llama-7b'
   'model path',
)


# Input prompt to continue.  Set NUM_EPOCHS=0 and RETOKENIZE=False to
# just generate continuations using the existing model.
_INPUT_PROMPT = flags.DEFINE_string(
   'INPUT_PROMPT',
   None,  # eg. 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?'
   'The initial "prompt" to continue generating from.',
)


# Currently set to 0 when running on a single GPU via xManager
_GPU = flags.DEFINE_integer('GPU', None, 'which gpu(s) to use')


# "Context length" of the model.
_MAX_LENGTH = flags.DEFINE_integer(
   'MAX_LENGTH', 512, 'Number of tokens to consider at once (ctx length)'
)

# Maximum generation length.
_MAX_GENERATION_LENGTH = flags.DEFINE_integer(
   'MAX_GENERATION_LENGTH', 1024, 'Number of tokens to generate at once'
)

# Training parameters.
_NUM_EPOCHS = flags.DEFINE_integer('NUM_EPOCHS', 500, 'number of training epochs') # number of epochs to train for
_BATCH_SIZE = flags.DEFINE_integer('BATCH_SIZE', 1, 'batch size') # batch size
_LEARNING_RATE = flags.DEFINE_float('LEARNING_RATE', 0.0001, 'learning rate') # learning rate
_WEIGHT_DECAY = flags.DEFINE_float('WEIGHT_DECAY', 0.0001, 'weight decay') # weight decay
# _DROPOUT = flags.DEFINE_float('DROPOUT', 0.0, 'dropout') # dropout

# compute rouge score
def rouge_compute(predictions, references, rouge_types=None, use_aggregator=True, use_stemmer=False):
  if rouge_types is None:
    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

  scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
  if use_aggregator:
    aggregator = scoring.BootstrapAggregator()
  else:
    scores = []

  for ref, pred in zip(references, predictions):
    score = scorer.score(ref, pred)
    if use_aggregator:
      aggregator.add_scores(score)
    else:
      scores.append(score)

  if use_aggregator:
    result = aggregator.aggregate()
  else:
    result = {}
    for key in scores[0]:
      result[key] = [score[key] for score in scores]

  return result

# Load model and tokenizer.
def load_model(model_path: str):
  cache_dir = tempfile.gettempdir()
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      "/cns/iq-d/home/kkaizh/Gemma/gemma-1.1-2b-it", cache_dir=cache_dir
  )
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.pad_token_id = tokenizer.eos_token_id
  tokenizer.padding_side = "left"
  model = transformers.AutoModelForCausalLM.from_pretrained(
      "/cns/iq-d/home/kkaizh/Gemma/gemma-1.1-2b-it", cache_dir=cache_dir
  )
  return model, tokenizer


# Convert model to DDP.
def model_to_train(model,
  device: torch.device,
  local_rank: int = 0,
  world_size: int = 1) -> transformers.models.gemma.modeling_gemma.GemmaForCausalLM | torch.nn.Module:
  # device = torch.device("cuda", local_rank)
  if world_size > 1:
    print("DDP model loading")
    logging.info(torch.cuda.device_count())
    ddp_model = DDP(model.to(device), device_ids=[local_rank])
    model2train = ddp_model.module
  else:
    print("Single GPU")
    model2train = model.to(device)
  return model2train


# Setup training arguments.
def get_training_args(
   num_epochs: int,
   batch_size: int,
   learning_rate: float,
   weight_decay: float,
) -> transformers.Seq2SeqTrainingArguments:
  return transformers.Seq2SeqTrainingArguments(
      warmup_steps=10,
      logging_steps=100,
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      num_train_epochs=num_epochs,
      logging_dir='/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/instruction_tuning/logs',
      output_dir='/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/instruction_tuning',
      per_device_eval_batch_size=batch_size,
      per_device_train_batch_size=batch_size,
      # evaluation_strategy="steps",
      # eval_accumulation_steps=1,
      # eval_steps=10000,
      save_strategy="no",
      # save_steps=10000,
      save_safetensors=False,
      # save_total_limit=1,
      load_best_model_at_end=True,
      # metric_for_best_model="rouge",
      report_to=[]
  )


# Setup trainer.
def get_trainer(model,tokenizer,
   training_args: transformers.Seq2SeqTrainingArguments,
   train_dataset,
   eval_dataset,
) -> transformers.Seq2SeqTrainer:
  
  # customized compute metrics
  def compute_metrics(eval_pred):
    print("Compute Metrics")
    prediction_logits, labels=eval_pred
    predictions = np.argmax(prediction_logits, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    print(decoded_preds)
    print(decoded_labels)
    result = rouge_compute(predictions=decoded_preds, references=decoded_labels)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}


    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    print(result)
    
    return {k: round(v, 4) for k, v in result.items()}
  
  return transformers.Seq2SeqTrainer(
      model=model,
      tokenizer=tokenizer,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      compute_metrics=compute_metrics,
      # callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)],
      data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]), 'attention_mask': torch.stack([f[1] for f in data]), 'labels': torch.stack([f[2] for f in data])}
   )


# Load modified dataset.
def load_raw_dataset(input_file_path: str):
  with gfile.Open(os.path.join(input_file_path, "modified_train.json"), "r", encoding="utf-8") as f:
    raw_train_data = json.load(f)
  with gfile.Open(os.path.join(input_file_path, "modified_test.json"), "r", encoding="utf-8") as f:
    raw_test_data = json.load(f)
  return raw_train_data, raw_test_data

# wrap with dataset class
class my_dataset(Dataset):
  def __init__(self, raw_data, tokenizer, max_length):
    self.raw_data = raw_data
    self.tokenizer = tokenizer
    self.max_length = max_length
    # self.max_target_length = max_length
    
  def __len__(self):
    return len(self.raw_data)
  
  def __getitem__(self, idx):
    data = self.raw_data[idx]
    # gsm8k
    # input_text = instruction_prompt.format(question=data['question']) # modified for dataset shift, for GSM8k -> Question/Answer; for WMT -> src/target
    # target_text = output.format(pCheck="Yes" if int(data["privacy"])==1 else "No", answer=data['answer'])
    # input_ = self.tokenizer(input_text,return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
    # input_ids = input_["input_ids"].squeeze(0)
    # attn_masks = input_["attention_mask"]
    # target = self.tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
    # labels = target["input_ids"]
    # wmt
    input_text = instruction_prompt_wmt.format(src=data['src']) # modified for dataset shift, for GSM8k -> Question/Answer; for WMT -> src/target
    target_text = output_wmt.format(target=data["target"])
    input_ = self.tokenizer(input_text,return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
    input_ids = input_["input_ids"].squeeze(0)
    attn_masks = input_["attention_mask"]
    target = self.tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
    labels = target["input_ids"]
    
    return input_ids, attn_masks, labels


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
   raise app.UsageError("Too many command-line arguments.")
  
  # Environment variables for Torch distributed training.
  # We can have multiple nodes, each with multiple GPUs.
  # Each node is a separate process, and each GPU is a separate device.
  # The global rank is the unique rank of the process across all nodes.
  torch.manual_seed(1337)
  os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
  # Rank of the current process on the current node.
  local_rank = int(os.environ['LOCAL_RANK'])
  # Number of workers per node.  For single GPU, this is 1.  For multi-GPU,
  # this is the number of GPUs per node.
  num_worker_per_node = int(os.environ['LOCAL_WORLD_SIZE'])
  # Rank of the current node.
  node_rank = int(os.environ['GROUP_RANK'])
  # Total number of nodes.
  world_size = int(os.environ['WORLD_SIZE'])
  logging.info(
    'local_rank=%s, num_worker_per_node=%s, node_rank=%s, world_size=%s',
    local_rank,
    num_worker_per_node,
    node_rank,
    world_size,
  )


  if torch.cuda.device_count() > 0:
    print("local_rank:{}".format(local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
  else:
    device = torch.device('cpu')


  if world_size > 1:
    global_rank = node_rank * num_worker_per_node + local_rank
    dist.init_process_group(
        backend='NCCL',
        rank=global_rank,
        world_size=world_size,
    )
  else:
    global_rank = 0
  logging.info('Using device: %s', device)
  # Load model and tokenizer.
  print("Loading model")
  print(_MODEL_PATH.value)
  model, tokenizer = load_model(_MODEL_PATH.value)
  print("Model loaded")
  # Load mpdel into GPU.
  model2train = model_to_train(model, device, local_rank, world_size)
  
  # define generation configuration
  generation_config = transformers.GenerationConfig(
        max_length=_MAX_GENERATION_LENGTH.value,
        renomalize_logits=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
  
  # Load dataset and processing.
  # raw_train_data, raw_eval_data = load_raw_dataset(_INPUT_FILE_PATH.value)
  # train_dataset = my_dataset(raw_train_data, tokenizer, _MAX_LENGTH.value)
  # # val_dataset = my_dataset(raw_eval_data[:10], tokenizer, _MAX_LENGTH.value)
  # eval_dataset = my_dataset(raw_eval_data, tokenizer, _MAX_LENGTH.value)
  
  # wmt22
  with gfile.Open(r"/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/datasets/wmt22/cs_en.json") as f:
    raw_data = json.load(f)
  wmt_train_dataset = my_dataset(raw_data[10000:20000], tokenizer, 256)
  wmt_eval_dataset = my_dataset(raw_data[:-1000], tokenizer, 256)
  
  # Setup trainer.
  training_args = get_training_args(
      num_epochs=_NUM_EPOCHS.value,
      batch_size=_BATCH_SIZE.value,
      learning_rate=_LEARNING_RATE.value,
      weight_decay=_WEIGHT_DECAY.value,
  )
  trainer = get_trainer(
      model=model2train,
      tokenizer=tokenizer,
      training_args=training_args,
      train_dataset=wmt_train_dataset,
      eval_dataset=wmt_eval_dataset,
  )
  torch.cuda.empty_cache()
  # Train.
  try:
    trainer.train()
    trainer.save_model(r"/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/instruction_tuning")
    torch.cuda.empty_cache()
  except Exception as e:
    print(e)
    print(torch.cuda.memory_summary())

  # starting generation
  generated_content = []
  
  # gsm8k
  # with tqdm.tqdm(total=len(raw_eval_data)) as pbar:
  #   pbar.set_description("Generating answers")
  #   for data in raw_eval_data:
  #     question = tokenizer(instruction_prompt_wmt.format(question=data['question']), return_tensors="pt", max_length=1536)
  #     prompt_length = question["input_ids"].shape[1]
  #     output = model2train.generate(question["input_ids"].to(device), generation_config=generation_config)
  #     transition_scores = model.compute_transition_scores(
  #         output.sequences, output.scores, normalize_logits=True
  #         )
  #     generated_tokens = output.sequences[:, prompt_length:]
  #     decoded_answer = tokenizer.decode(generated_tokens[0])
  #     raw_logits = np.exp(transition_scores.cpu().numpy())
  #     mean_logits = np.mean(raw_logits)
  #     median_logits = np.median(raw_logits)
  #     quantile25 = np.quantile(raw_logits, 0.25)
  #     quantile50 = np.quantile(raw_logits, 0.50)
  #     quantile75 = np.quantile(raw_logits, 0.75)
  #     logits = {"mean": str(mean_logits), "median": str(median_logits), "quantile25": str(quantile25), "quantile50": str(quantile50), "quantile75": str(quantile75)}
  #     generated_content.append({"src": data['src'], "target": decoded_answer, "logits": logits})
  #     pbar.update(10)
  
  # wmt
  with tqdm.tqdm(total=len(raw_data[:10000])) as pbar:
    pbar.set_description("Generating answers")
    for data in raw_data[:10000]:
      question = tokenizer(instruction_prompt_wmt.format(src=data['src']), return_tensors="pt", max_length=1536)
      prompt_length = question["input_ids"].shape[1]
      output = model2train.generate(question["input_ids"].to(device), generation_config=generation_config)
      transition_scores = model.compute_transition_scores(
          output.sequences, output.scores, normalize_logits=True
          )
      generated_tokens = output.sequences[:, prompt_length:]
      decoded_answer = tokenizer.decode(generated_tokens[0])
      rouge_score = rouge_compute(decoded_answer, data["target"])
      raw_logits = np.exp(transition_scores.cpu().numpy())
      mean_logits = np.mean(raw_logits)
      median_logits = np.median(raw_logits)
      quantile25 = np.quantile(raw_logits, 0.25)
      quantile50 = np.quantile(raw_logits, 0.50)
      quantile75 = np.quantile(raw_logits, 0.75)
      logits = {"mean": str(mean_logits), "median": str(median_logits), "quantile25": str(quantile25), "quantile50": str(quantile50), "quantile75": str(quantile75)}
      generated_content.append({"src": data['src'], "target": decoded_answer, "logits": logits, "rouge": rouge_score})
      pbar.update(10)
  
  # store results    
  # try:
  #   with gfile.Open("/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/datasets/wmt22/instruct_results.json", "w", encoding="utf-8") as f:
  #     json.dump(generated_content, f)
  # except Exception as e:
  #   print(e)



if __name__ == "__main__":
 g3_multiprocessing.handle_main(gdist.torchrun(main))
