from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging

from google3.pyglib import gfile

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# local_rank = int(os.environ["LOCAL_RANK"])

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

from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

gemma2b_path = r"/cns/iq-d/home/kkaizh/Gemma/gemma-1.1-2b-it"
gemma7b_path = r"/cns/iq-d/home/kkaizh/Gemma/gemma-1.1-7b-it"

class MultiTaskModel(torch.nn.Module):
    def __init__(self, local_model, server_model, threshold):
        super().__init__()
        cache_dir = tempfile.gettempdir()
        self.threshold = threshold
        self.gemma2b_causal_lm = transformers.AutoModelForCausalLM.from_pretrained(local_model, cache_dir=cache_dir)
        self.gemma2b_sequence_classifier = transformers.AutoModelForSequenceClassification.from_pretrained(local_model, cache_dir=cache_dir)
        self.gemma7b_causal_lm = transformers.AutoModelForCausalLM.from_pretrained(server_model, output_hidden_states=True, cache_dir=cache_dir)  # Get hidden states for distillation

    def forward(self, input_ids, attention_mask, labels=None, binary_labels=None):
        causal_lm_outputs = self.gemma2b_causal_lm(input_ids, attention_mask)
        sequence_classifier_outputs = self.gemma2b_sequence_classifier(input_ids, attention_mask)
        with torch.no_grad():
            gemma7b_causal_lm_outputs = self.gemma7b_causal_lm(input_ids, attention_mask)
            gemma7b_hidden_states = gemma7b_causal_lm_outputs.hidden_states[-1]  # Last hidden state for distillation

        # Distillation loss: Calculate cosine similarity between Gemma-2b and Gemma-7b hidden states
        distillation_loss = torch.mean(torch.cosine_similarity(causal_lm_outputs.hidden_states[-1], gemma7b_hidden_states, dim=-1))

        # Combined loss: Weighted sum of causal LM loss, sequence classifier loss, and distillation loss
        causal_lm_loss = causal_lm_outputs.loss
        sequence_classifier_loss = sequence_classifier_outputs.loss
        
        if torch.mean(causal_lm_outputs.logits) > self.threshold:
          combined_loss = 0.6*causal_lm_loss + 0.4*sequence_classifier_loss
        else:
          combined_loss = 0.5*causal_lm_loss + 0.4*sequence_classifier_loss + 0.1*causal_lm_loss*distillation_loss

        return combined_loss #, causal_lm_outputs.logits, sequence_classifier_outputs.logits

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
    
def compute_metrics(eval_pred):
    text_generation_preds, text_generation_labels = eval_pred[0], eval_pred[1]
    binary_classification_preds, binary_classification_labels = eval_pred[2], eval_pred[3]

    '''
    modify the metric computing
    '''
    text_generation_rouge = rouge_compute(text_generation_preds, text_generation_labels)
    binary_classification_acc = (binary_classification_preds.argmax(dim=1) == binary_classification_labels).mean()

    return {
        'rouge-l': text_generation_rouge['rouge-l'][0]['f'],
        'accuracy': binary_classification_acc
    }


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
  
  training_args = transformers.TrainingArguments(
        output_dir='./results',
        evaluation_strategy='steps',
        eval_steps=100,
        save_steps=100,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        load_best_model_at_end=True,
        metric_for_best_model='rouge-l',
    )
  tokenizer = transformers.AutoTokenizer.from_pretrained(gemma2b_path)
  model = MultiTaskModel(gemma2b_path, gemma7b_path, 0.6)
  
  # dataset loading
  
  
  trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
  # training
  trainer.train()
  
if __name__ == "__main__":
 g3_multiprocessing.handle_main(gdist.torchrun(main))