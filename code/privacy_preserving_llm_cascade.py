instruction_prompt = r'''Assume you're a professional student working on mathematical problems. Now, you'll be given mathematical problems, please follow the pattern in the examples to solve the tasks below: 
a. Check if the question contains personal information (e.g., names etc.), output Yes or No only;\n
b. Solve this question; \n

Herer are the example:
Question: Hector purchased a container of gumballs. He gave 4 to Todd, then he gave twice as many as he had given Todd to Alisha, and then he gave 5 less than four times as many to Bobby as he had given to Alisha. If Hector had 6 gumballs remaining, what is the total number of gumballs that Hector purchased? \n
Output: 
Let's think step by step:
For a, the question contains personal names so the answer is Yes.
For b, Hector gave to Alisha twice as many as he had given Todd, for a total of 4*2=<<4*2=8>>8 gumballs, Hector gave 5 less than four times as many to Bobby as he had given to Alisha, or a total of (8*4)-5=<<8*4-5=27>>27 gumballs. If Hector had 6 gumballs remaining, he originally purchased 4+8+27+6=<<4+8+27+6=45>>45 gumballs.
a. Contains Personal Information: Yes.
b. Answer: 45.

Case,
Question: {question}\n
Output:
Let's step by step:
'''


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load local and server models/tokenizers
local_model_name = "gemma2-2b"
server_model_name = "gemma2-9b"
local_tokenizer = AutoTokenizer.from_pretrained(local_model_name)
local_model = AutoModelForCausalLM.from_pretrained(local_model_name)
server_tokenizer = AutoTokenizer.from_pretrained(server_model_name)
server_model = AutoModelForCausalLM.from_pretrained(server_model_name)

# Private memory (a growing dictionary for storing private tokens)
private_memory = {}

# Mock dataset class
class QueryDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = instruction_prompt.format(question=self.data[idx]['question'])
        output_text = self.data[idx]['answer']
        encoded = self.tokenizer(
            input_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        labels = self.tokenizer(
            output_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        encoded["labels"] = labels["input_ids"]
        return {key: val.squeeze(0) for key, val in encoded.items()}

# Deferral logic using chain-of-thoughts

def chain_of_thoughts_deferral(local_logits, local_output, quality_threshold=0.5):
    """Chain-of-thoughts logic for deferral decision-making."""
    # Step 1: Quality check
    probs = torch.softmax(local_logits, dim=-1)
    max_prob, _ = torch.max(probs, dim=-1)
    quality_accept = max_prob >= quality_threshold

    if quality_accept:
        return "accept", local_output  # Accept the local answer

    # Step 2: Privacy judgment
    private_tokens = []
    if "name" in local_output.lower():
        private_tokens.append("name")
        private_memory[len(private_memory)] = "name"

    if private_tokens:
        return "privacy_deferral", private_tokens  # Contains private information

    return "quality_deferral", None  # Quality issue without privacy concern

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
)

# Custom loss function
def compute_loss(model, inputs):
    """Custom loss function including quality, privacy, and distillation losses."""
    outputs = model(**inputs)
    local_logits = outputs.logits

    # Extracting labels
    labels = inputs["labels"]

    # Quality loss (answer correctness)
    quality_loss = torch.nn.CrossEntropyLoss()(local_logits.view(-1, local_logits.size(-1)), labels.view(-1))

    # Privacy judgment loss (binary classification loss)
    privacy_labels = inputs.get("privacy_labels", None)
    if privacy_labels is not None:
        privacy_logits = local_logits[:, -1, :]  # Assuming the last token is used for privacy judgment
        privacy_loss = torch.nn.BCEWithLogitsLoss()(privacy_logits, privacy_labels)
    else:
        privacy_loss = 0

    # Distillation loss
    distillation_loss = 0
    if inputs.get("use_distillation", False):
        with torch.no_grad():
            masked_input = inputs["masked_input"]
            server_outputs = server_model(**masked_input)
            server_logits = server_outputs.logits

        distillation_loss = torch.nn.MSELoss()(local_logits, server_logits)

    # Combine losses
    total_loss = quality_loss + privacy_loss + distillation_loss
    return total_loss

# Trainer setup
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Chain-of-thoughts deferral process
        outputs = model(**inputs)
        local_logits = outputs.logits
        local_output = local_tokenizer.decode(torch.argmax(local_logits, dim=-1))

        decision, context = chain_of_thoughts_deferral(local_logits, local_output)

        if decision == "accept":
            loss = torch.nn.CrossEntropyLoss()(local_logits.view(-1, local_logits.size(-1)), inputs["labels"].view(-1))
        elif decision == "privacy_deferral":
            loss = torch.nn.BCEWithLogitsLoss()(local_logits[:, -1, :], inputs["privacy_labels"])
        else:  # quality_deferral
            masked_input = inputs["masked_input"]
            server_outputs = server_model(**masked_input)
            server_logits = server_outputs.logits
            loss = torch.nn.MSELoss()(local_logits, server_logits)

        return (loss, outputs) if return_outputs else loss

# Load dataset (mock data for demonstration)
data = [
    {"input": "Assume you’re a student...Question: Hector purchased...", "output": "Contains Personal Information: No\nAnswer: 45"},
    {"input": "Assume you’re a student...Question: Every time Carl...", "output": "Contains Personal Information: Yes\nAnswer: 6"},
]
train_dataset = QueryDataset(data, local_tokenizer)

# Initialize trainer
trainer = CustomTrainer(
    model=local_model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=local_tokenizer,
)

# Training
trainer.train()
