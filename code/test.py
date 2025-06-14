"""TODO(kkaizh): DO NOT SUBMIT without one-line documentation for test.

TODO(kkaizh): DO NOT SUBMIT without a detailed description of test.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging

import transformers
import tempfile
import json
import numpy as np
import nltk
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# import datasets
from google3.third_party.google_research.google_research.rouge import rouge_scorer
from google3.third_party.google_research.google_research.rouge import scoring

import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.google import distributed as gdist
from torch.nn.parallel import DistributedDataParallel as DDP

from google3.pyglib import gfile
from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

# Train on GSM8K dataset.
# Input file can be either a CNS or google3 file path.
# Default for XManager GPU training is train.jsonl.
_INPUT_FILE_PATH = flags.DEFINE_string(
   'INPUT_FILE_PATH',
   '/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/datasets/GSM8K',
   'input file path',
)


# Output folder for CSV of tokenized training pairs and PyTorch model.
_OUTPUT_FOLDER_PATH = flags.DEFINE_string(
   'OUTPUT_FOLDER_PATH',
   '/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/GSM8K/output/',  # eg. '/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/GSM8K/output/'
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
   'MAX_LENGTH', 1024, 'Number of tokens to consider at once (ctx length)'
)


# Training parameters.
_NUM_EPOCHS = flags.DEFINE_integer('NUM_EPOCHS', 1, 'number of training epochs') # number of epochs to train for
_BATCH_SIZE = flags.DEFINE_integer('BATCH_SIZE', 1, 'batch size') # batch size
_LEARNING_RATE = flags.DEFINE_float('LEARNING_RATE', 0.0001, 'learning rate') # learning rate
_WEIGHT_DECAY = flags.DEFINE_float('WEIGHT_DECAY', 0.0001, 'weight decay') # weight decay
# _DROPOUT = flags.DEFINE_float('DROPOUT', 0.0, 'dropout') # dropout

# rouge metric

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

class my_dataset(Dataset):
  def __init__(self, raw_data, tokenizer, max_length):
    self.raw_data = raw_data
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.max_target_length = max_length
    self.input_ids = []
    self.attn_masks = []
    self.labels = []
    
    for data in self.raw_data:
      input_text = data['question']
      target_text = data['answer']
      input_ = self.tokenizer(input_text,return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
      self.input_ids.append(input_["input_ids"].squeeze(0))
      self.attn_masks.append(input_["attention_mask"])
      target = self.tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_target_length)
      self.labels.append(target["input_ids"].squeeze(0))
    
  def __len__(self):
    return len(self.raw_data)
  
  def __getitem__(self, idx):
    # data = self.raw_data[idx]
    # input_text = data['question']
    # target_text = data['answer']
    # input_ = self.tokenizer(input_text,return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
    # input_ids = input_["input_ids"]
    # print(input_ids.shape)
    # attn_masks = input_["attention_mask"]
    # target = self.tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
    # labels = target["input_ids"]
    # print(labels.shape)
    return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]
    
    # return input_ids, attn_masks, labels

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
  
  model_path = '/cns/iq-d/home/kkaizh/Gemma/gemma-1.1-2b-it' # gemma-1.1-2b-it
  cache_dir = tempfile.gettempdir()
  model = transformers.AutoModelForCausalLM.from_pretrained(
      model_path,
      cache_dir=cache_dir
  )
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_path,
      cache_dir=cache_dir
  )
  
  # Load mpdel into GPU.
  model2train = model_to_train(model, device, local_rank, world_size)
  
  # loading datasets
  input_file_path = '/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/datasets/GSM8K'
  with gfile.Open(os.path.join(input_file_path, "train.jsonl"), "r", encoding="utf-8") as f:
    raw_train_data = [json.loads(line) for line in f]
  with gfile.Open(os.path.join(input_file_path, "test.jsonl"), "r", encoding="utf-8") as f:
    raw_test_data = [json.loads(line) for line in f]
  
  train_dataset = my_dataset(raw_train_data[:3], tokenizer, 128)
  test_dataset = my_dataset(raw_test_data[:2], tokenizer, 128)
  
  # test
  test_q = raw_test_data[0]['question']
  test_input = tokenizer(test_q, return_tensors="pt", max_length=128)
  prompt_length = test_input["input_ids"].shape[1]
  test_a = model2train.generate(test_input["input_ids"].to(device), max_length=512)
  print(tokenizer.decode(test_a[0][prompt_length:]))
  
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
  
  training_args = transformers.Seq2SeqTrainingArguments(
      warmup_steps=10,
      logging_steps=100,
      learning_rate=0.001,
      weight_decay=0.001,
      num_train_epochs=500,
      logging_dir='/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/instruction_tuning/logs',
      output_dir='/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/instruction_tuning',
      per_device_eval_batch_size=1,
      per_device_train_batch_size=1,
      evaluation_strategy="steps",
      eval_steps=5000,
      save_strategy="steps",
      save_steps=5000,
      save_safetensors=False,
      save_total_limit=1,
      load_best_model_at_end=True,
      metric_for_best_model="rouge",
      report_to=[]
  )
    
  trainer = transformers.Seq2SeqTrainer(
      model=model2train,
      tokenizer=tokenizer,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      # callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)], # enable if you increa the epoches
      data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]), 'attention_mask': torch.stack([f[1] for f in data]), 'labels': torch.stack([f[2] for f in data])},
      compute_metrics=compute_metrics,
  )
  torch.cuda.empty_cache()
  print("training")
  print(trainer.train())
  test_a_trained = model2train.generate(test_input["input_ids"].to(device), max_length=512)
  print("---------Printing Trained response------")
  print(tokenizer.decode(test_a_trained[0][prompt_length:]))
  # trainer.save_model('/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/instruction_tuning')
  # torch.cuda.empty_cache()
  # print("Evaluating")
  # print(trainer.evaluate())
  # print("Done")
  # trainer.push_to_hub()
    
if __name__ == "__main__":
  # app.run(main)
  g3_multiprocessing.handle_main(gdist.torchrun(main))
