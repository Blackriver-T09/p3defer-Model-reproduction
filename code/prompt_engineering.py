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
# local_rank = int(os.environ["LOCAL_RANK"])


# from config import get_opt


import torch
import torch.distributed as dist
from torch.google import distributed as gdist
# from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP


# from transformers import pipeline
import transformers
import tempfile
import numpy as np
# import nltk
import json
import tqdm
# import datasets

from google3.third_party.google_research.google_research.rouge import rouge_scorer
from google3.third_party.google_research.google_research.rouge import scoring

# from google3.pyglib import gfile
from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

# prompt tuning for cascading, two-shot prompting
inference_prompt_2_shot = r'''Assume you're a student working on some mathematical problems. Now, you'll be giving mathematical problems, you need to do four tasks: a. Check if the question contains personal information (e.g., names etc.), output "Contains personal information: Yes or No only";\n
b. Solve this question; \n

Here are some examples:
Question: Hector purchased a container of gumballs. He gave 4 to Todd, then he gave twice as many as he had given Todd to Alisha, and then he gave 5 less than four times as many to Bobby as he had given to Alisha. If Hector had 6 gumballs remaining, what is the total number of gumballs that Hector purchased? \n
Output: 
a. Contains Personal Information: Yes
b. Answer: Hector gave to Alisha twice as many as he had given Todd, for a total of 4*2=<<4*2=8>>8 gumballs, Hector gave 5 less than four times as many to Bobby as he had given to Alisha, or a total of (8*4)-5=<<8*4-5=27>>27 gumballs. If Hector had 6 gumballs remaining, he originally purchased 4+8+27+6=<<4+8+27+6=45>>45 gumballs. #### 45

Question: Krystian works in the library. He borrows an average of 40 books every day. Every Friday, his number of borrowed books is about 40% higher than the daily average. How many books does he borrow in a week if the library is open from Monday to Friday? \n
Output: 
a. Contains Personal Information: Yes
b. Answer: The number of books borrowed on Friday is higher by 40 * 40/100 = <<40*40/100=16>>16 books. There are 5 days from Monday to Friday inclusive, so Krystian borrows an average of 5 * 40 = <<5*40=200>>200 books during that time. With Friday's increase in borrowings, during one week Krystian borrows 200 + 16 = <<200+16=216>>216 books. #### 216

Your turn,
Question: {question}\n
Output:'''

inference_prompt_5_shot = r"""Assume you're a student working on some mathematical problems. Now, you'll be giving mathematical problems, you need to do three tasks: a. Check if the question contains personal information (e.g., names etc.), output "Contains personal information: Yes or No only";\n
b. Solve this question; \n

Here are some examples:
Question: Hector purchased a container of gumballs. He gave 4 to Todd, then he gave twice as many as he had given Todd to Alisha, and then he gave 5 less than four times as many to Bobby as he had given to Alisha. If Hector had 6 gumballs remaining, what is the total number of gumballs that Hector purchased? \n
Output: 
a. Contains Personal Information: Yes
b. Answer: Hector gave to Alisha twice as many as he had given Todd, for a total of 4*2=<<4*2=8>>8 gumballs, Hector gave 5 less than four times as many to Bobby as he had given to Alisha, or a total of (8*4)-5=<<8*4-5=27>>27 gumballs. If Hector had 6 gumballs remaining, he originally purchased 4+8+27+6=<<4+8+27+6=45>>45 gumballs. #### 45.

Question: A garden produced 237 potatoes, 60 fewer cucumbers and twice as many peppers than the cucumbers. How many vegetables did the garden produce? \n
Output:
a. Contains Personal Information: No
b. Answer: The garden produced 237 potatoes - 60 = <<237-60=177>>177 cucumbers. The garden produced 177 cucumbers * 2 peppers/cucumber = <<177*2=354>>354 peppers. The garden produced 237 potatoes + 177 cucumbers + 354 peppers = <<237+177+354=768>>768 vegetables. #### 768

Question: A boxer weighs 97 kg at 4 months from a fight. He is on a diet that allows him to lose 3 kg per month until the day of the fight. How much will he weigh on the day of the fight?
Output: 
a. Contains Personal Information: No
b. Answer: In 4 months, he will lose 3 x 4 = <<3*4=12>>12 kilograms. So his weight will be 97 – 12 = <<97-12=85>>85 kilograms. #### 85

Question: Krystian works in the library. He borrows an average of 40 books every day. Every Friday, his number of borrowed books is about 40% higher than the daily average. How many books does he borrow in a week if the library is open from Monday to Friday? \n
Output: 
a. Contains Personal Information: Yes
b. Answer: The number of books borrowed on Friday is higher by 40 * 40/100 = <<40*40/100=16>>16 books. There are 5 days from Monday to Friday inclusive, so Krystian borrows an average of 5 * 40 = <<5*40=200>>200 books during that time. With Friday's increase in borrowings, during one week Krystian borrows 200 + 16 = <<200+16=216>>216 books. #### 216

Question: Sally and Bob have made plans to go on a trip at the end of the year. They both decide to work as babysitters and save half of what they've earned for their trip. If Sally makes $6 per day and Bob makes $4 per day, how much money will they both have saved for their trip after a year? \n
Output:
a. Contains Personal Information: Yes
b. Answer: Saly saves 1/2 * $6/day = $<<1/2*6=3>>3/day. Since each year have 365 days, the total amount of money Sally will save in a year is $3/day * 365 days/year = $<<3*365=1095>>1095/year Bob saves 1/2 * $4/day = $<<1/2*4=2>>2/day. The total amount of money Bob will have saved in a year is $2/day * 365 days/year = $<<2*365=730>>730/year In total, Sally and Bob would have saved $730 + $1095 = $<<730+1095=1825>>1825 #### 1825

Your turn:
Question: {question}\n
Output:"""

# prompt for no cascading
inference_prompt = r'''Please answer a mathematical question and output your answer following this format: <Solution ### Final Answer>\n
For example:
Question - Hector purchased a container of gumballs. He gave 4 to Todd, then he gave twice as many as he had given Todd to Alisha, and then he gave 5 less than four times as many to Bobby as he had given to Alisha. If Hector had 6 gumballs remaining, what is the total number of gumballs that Hector purchased?\n
Answer - Hector gave to Alisha twice as many as he had given Todd, for a total of 4*2=<<4*2=8>>8 gumballs, Hector gave 5 less than four times as many to Bobby as he had given to Alisha, or a total of (8*4)-5=<<8*4-5=27>>27 gumballs. If Hector had 6 gumballs remaining, he originally purchased 4+8+27+6=<<4+8+27+6=45>>45 gumballs. #### 45. \n

Question - Sally and Bob have made plans to go on a trip at the end of the year. They both decide to work as babysitters and save half of what they've earned for their trip. If Sally makes $6 per day and Bob makes $4 per day, how much money will they both have saved for their trip after a year? \n
Answer - Saly saves 1/2 * $6/day = $<<1/2*6=3>>3/day. Since each year have 365 days, the total amount of money Sally will save in a year is $3/day * 365 days/year = $<<3*365=1095>>1095/year Bob saves 1/2 * $4/day = $<<1/2*4=2>>2/day. The total amount of money Bob will have saved in a year is $2/day * 365 days/year = $<<2*365=730>>730/year In total, Sally and Bob would have saved $730 + $1095 = $<<730+1095=1825>>1825 #### 1825\n

Case:
Question - {question} \n
Answer - '''

# prompt for translating
prompt_translate_0shot=r'''
Assume you're a professional translator, now please translate the following sentence into English. Output the translated sentence only.\n
Source sentence: {src}\n
Target sentence:
'''

prompt_translate_2shot=r'''
Assume you're a professional translator, now please translate the following sentence into English. Output the translated sentence only.\n

For example:
Source sentence: Následný postup na základě usnesení Parlamentu: viz zápis\n
Target sentence: Action taken on Parliament's resolutions: see Minutes\n

Source sentence: zprávě paní Marie Grazie Paganové předložené jménem Výboru pro občanské svobody, spravedlnost a vnitřní věci o rozvoji v oblasti trestního soudnictví EU\n
Target sentence: the report by Maria Grazia Pagano, on behalf of Committee on Civil Liberties, Justice and Home Affairs, on development of an EU criminal justice area, with a proposal for a European Parliament recommendation to the Council on development of an EU criminal justice area.\n

Case:
Source sentence: {src}
Target sentence:
'''

prompt_translate_5shot=r'''
Assume you're a professional translator, now please translate the following sentence into English. Output the translated sentence only.\n

For example:
Source sentence: Následný postup na základě usnesení Parlamentu: viz zápis\n
Target sentence: Action taken on Parliament's resolutions: see Minutes\n

Source sentence: zprávě paní Marie Grazie Paganové předložené jménem Výboru pro občanské svobody, spravedlnost a vnitřní věci o rozvoji v oblasti trestního soudnictví EU\n
Target sentence: the report by Maria Grazia Pagano, on behalf of Committee on Civil Liberties, Justice and Home Affairs, on development of an EU criminal justice area, with a proposal for a European Parliament recommendation to the Council on development of an EU criminal justice area.\n

Source sentence: Přestože na začátku jednání byla naše stanoviska od sebe dosti vzdálená, podařilo se nám dosáhnout kompromisu, který nám, jak doufám, umožní, abychom v Radě dospěli ke shodě při prvním čtení.\n
Target sentence: In spite of our fairly distant initial negotiating positions, we have managed to reach a compromise, which, I hope, will allow us to come to an understanding with the Council at first reading.\n

Source sentence: Navrženým nástrojem se zřizuje zvláštní postup.\n
Target sentence: The proposed instrument establishes a special procedure.\n

Source sentence: V navrženém postupu jsem se nicméně snažil dosáhnout co možná největší pružnosti a zkrácení lhůt určených pro odpovědi Komise a rovněž snížení byrokratické zátěže.\n
Target sentence:However, I have tried to achieve the greatest possible flexibility in the proposed procedure and a shortening of the times designated for the Commission to react, and also a reduction in the bureaucratic load.\n

Case:
Source sentence: {src}
Target sentence:
'''

# model capability
inference_prompt_0_shot = r'''Please answer the given mathematical question: {question}.'''

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
   '/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/prompt_tuning/',  # eg. '/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/GSM8K/output/'
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


# Load model and tokenizer.
def load_model(model_path: str):
  cache_dir = tempfile.gettempdir()
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      "/cns/iq-d/home/kkaizh/Gemma/gemma-1.1-7b-it", cache_dir=cache_dir
  )
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.pad_token_id = tokenizer.eos_token_id
  tokenizer.padding_side = "left"
  model = transformers.AutoModelForCausalLM.from_pretrained(
      "/cns/iq-d/home/kkaizh/Gemma/gemma-1.1-7b-it", cache_dir=cache_dir
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

# Load dataset.
def load_raw_dataset(input_file_path: str):
  with gfile.Open(os.path.join(input_file_path, "train.jsonl"), "r", encoding="utf-8") as f:
    raw_train_data = [json.loads(line) for line in f]
  with gfile.Open(os.path.join(input_file_path, "test.jsonl"), "r", encoding="utf-8") as f:
    raw_test_data = [json.loads(line) for line in f]
  with gfile.Open(os.path.join(input_file_path, "train_socratic.jsonl"), "r", encoding="utf-8") as f:
    raw_train_socratic_data = [json.loads(line) for line in f]
  with gfile.Open(os.path.join(input_file_path, "test_socratic.jsonl"), "r", encoding="utf-8") as f:
    raw_test_socratic_data = [json.loads(line) for line in f]
  return raw_train_data, raw_test_data, raw_train_socratic_data, raw_test_socratic_data

# rouge_compute
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



def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
   raise app.UsageError("Too many command-line arguments.")
  
  print("Transformers version:{0}".format(transformers.__version__))
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
  
  # generation configuration for obtain logits
  generation_config = transformers.GenerationConfig(
        max_length=1024,
        renomalize_logits=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )

  # Adding an inference block
  generated_content_train = []
  generated_content_eval_2shot = []
  generated_content_eval_5shot = []
  generated_content_eval_0shot = []
  generated_content_socratic_train = []
  generated_content_socratic_eval = []
  # raw_train_data, raw_eval_data, raw_train_socratic_data, raw_eval_socratic_data = load_raw_dataset(_INPUT_FILE_PATH.value)
  with gfile.Open(r"/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/datasets/wmt22/cs_en.json", "r", encoding="utf-8") as f:
    raw_eval_data_ = json.load(f)
  raw_eval_data = raw_eval_data_[:10000]
  # obtain from train
  # with tqdm.tqdm(total=len(raw_train_data)) as pbar:
  #   pbar.set_description("Generating answers")
  #   for data in raw_train_data:
  #     question = tokenizer(inference_prompt_5_shot.format(question=data['question']), return_tensors="pt", max_length=1536)
  #     answer = model2train.generate(question["input_ids"].to(device), max_length=2048)
  #     prompt_length = question["input_ids"].shape[1]
  #     decoded_answer = tokenizer.decode(answer[0][prompt_length:])
  #     generated_content_train.append({"question": data['question'], "answer": decoded_answer})
  #     pbar.update(10)
  # print("length of raw data:{0}".format(len(raw_train_data)))
  # print("length of generated content:{0}".format(len(generated_content_train)))
  
  # with tqdm.tqdm(total=len(raw_eval_data)) as pbar:
  #   pbar.set_description("Generating answers")
  #   for data in raw_eval_data:
  #     question = tokenizer(prompt_translate_0shot.format(src=data['src']), return_tensors="pt", max_length=512) # 1536 for GSM8k
  #     prompt_length = question["input_ids"].shape[1]
  #     output = model2train.generate(question["input_ids"].to(device), generation_config=generation_config)
  #     transition_scores = model.compute_transition_scores(
  #         output.sequences, output.scores, normalize_logits=True
  #         )
  #     generated_tokens = output.sequences[:, prompt_length:]
  #     decoded_answer = tokenizer.decode(generated_tokens[0])
  #     rouge_score = rouge_compute(decoded_answer, data["target"])
  #     raw_logits = np.exp(transition_scores.cpu().numpy())
  #     mean_logits = np.mean(raw_logits)
  #     median_logits = np.median(raw_logits)
  #     quantile25 = np.quantile(raw_logits, 0.25)
  #     quantile50 = np.quantile(raw_logits, 0.50)
  #     quantile75 = np.quantile(raw_logits, 0.75)
  #     logits = {"mean": str(mean_logits), "median": str(median_logits), "quantile25": str(quantile25), "quantile50": str(quantile50), "quantile75": str(quantile75)}
  #     generated_content_eval_0shot.append({"src": data['src'], "answer": decoded_answer, "logits": logits, "rouge": rouge_score})
  #     pbar.update(10)
      
  with tqdm.tqdm(total=len(raw_eval_data)) as pbar:
    pbar.set_description("Generating answers")
    for data in raw_eval_data:
      question = tokenizer(prompt_translate_2shot.format(src=data['src']), return_tensors="pt", max_length=512) # 1536 for GSM8k
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
      generated_content_eval_2shot.append({"src": data['src'], "answer": decoded_answer, "logits": logits, "rouge": rouge_score})
      pbar.update(10)
    
  # with tqdm.tqdm(total=len(raw_eval_data)) as pbar:
  #   pbar.set_description("Generating answers")
  #   for data in raw_eval_data:
  #     question = tokenizer(prompt_translate_5shot.format(src=data['src']), return_tensors="pt", max_length=768) # 1536 for GSM8k
  #     prompt_length = question["input_ids"].shape[1]
  #     output = model2train.generate(question["input_ids"].to(device), generation_config=generation_config)
  #     transition_scores = model.compute_transition_scores(
  #         output.sequences, output.scores, normalize_logits=True
  #         )
  #     generated_tokens = output.sequences[:, prompt_length:]
  #     decoded_answer = tokenizer.decode(generated_tokens[0])
  #     rouge_score = rouge_compute(decoded_answer, data["target"])
  #     raw_logits = np.exp(transition_scores.cpu().numpy())
  #     mean_logits = np.mean(raw_logits)
  #     median_logits = np.median(raw_logits)
  #     quantile25 = np.quantile(raw_logits, 0.25)
  #     quantile50 = np.quantile(raw_logits, 0.50)
  #     quantile75 = np.quantile(raw_logits, 0.75)
  #     logits = {"mean": str(mean_logits), "median": str(median_logits), "quantile25": str(quantile25), "quantile50": str(quantile50), "quantile75": str(quantile75)}
  #     generated_content_eval_5shot.append({"src": data['src'], "answer": decoded_answer, "logits": logits, "rouge": rouge_score})
  #     pbar.update(10)
      
  # with tqdm.tqdm(total=len(raw_train_socratic_data)) as pbar:
  #   pbar.set_description("Generating answers")
  #   for data in raw_train_socratic_data:
  #     question = tokenizer(inference_prompt_5_shot.format(question=data['question']), return_tensors="pt", max_length=1536)
  #     answer = model2train.generate(question["input_ids"].to(device), max_length=2048)
  #     prompt_length = question["input_ids"].shape[1]
  #     decoded_answer = tokenizer.decode(answer[0][prompt_length:])
  #     generated_content_socratic_train.append({"question": data['question'], "answer": decoded_answer})
  #     pbar.update(10)
  
  # with tqdm.tqdm(total=len(raw_eval_socratic_data)) as pbar:
  #   pbar.set_description("Generating answers")
  #   for data in raw_eval_socratic_data:
  #     question = tokenizer(inference_prompt_5_shot.format(question=data['question']), return_tensors="pt", max_length=1536)
  #     answer = model2train.generate(question["input_ids"].to(device), max_length=2048)
  #     prompt_length = question["input_ids"].shape[1]
  #     decoded_answer = tokenizer.decode(answer[0][prompt_length:])
  #     generated_content_socratic_eval.append({"question": data['question'], "answer": decoded_answer})
  #     pbar.update(10)
  
  # try:
  #   with gfile.Open("/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/prompt_tuning/prompt_binary_train_5_shot.json", "w", encoding="utf-8") as f:
  #     json.dump(generated_content_train, f)
  # except Exception as e:
  #   print(e)
  # try:
  #   with gfile.Open("/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/prompt_tuning/cs_en_5_shot.json", "w", encoding="utf-8") as f:
  #     json.dump(generated_content_eval_5shot, f)
  # except Exception as e:
  #   print(e)
  
  try:
    with gfile.Open("/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/datasets/wmt22/cs_en_server.json", "w", encoding="utf-8") as f:
      json.dump(generated_content_eval_2shot, f)
  except Exception as e:
    print(e)
   
  # try:
  #   with gfile.Open("/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/prompt_tuning/cs_en_eval_0_shot.json", "w", encoding="utf-8") as f:
  #     json.dump(generated_content_eval_0shot, f)
  # except Exception as e:
  #   print(e) 
    
  # try:
  #   with gfile.Open("/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/prompt_tuning/prompt_binary_train_socratic_5_shot.json", "w", encoding="utf-8") as f:
  #     json.dump(generated_content_socratic_train, f)
  # except Exception as e:
  #   print(e)
  # try:
  #   with gfile.Open("/cns/iq-d/home/kkaizh/multi_obj_cascade_llm/prompt_tuning/prompt_binary_eval_socratic_5_shot.json", "w", encoding="utf-8") as f:
  #     json.dump(generated_content_socratic_eval, f)
  # except Exception as e:
    print(e)




if __name__ == "__main__":
 g3_multiprocessing.handle_main(gdist.torchrun(main))
