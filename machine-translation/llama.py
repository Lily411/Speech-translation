from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraModel, LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from datasets import Dataset, concatenate_datasets
import sys
import time
import os
import wandb

# ----------------------------------------------------------------------------------------------
wandb.login(key="422af8259e8566f1d41a435099d066bf3fb3a5d7")  # 替換 YOUR_API_KEY 為你的 API 密鑰

model_id = 'meta-llama/Meta-Llama-3-8B'
token = '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' # 替換為你的 Token

tokenizers = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
tokenizers.pad_token_id = 0
tokenizers.padding_size = ' right'


device = 'cpu'
if torch.cuda.is_available():
    device = "cuda"

t0 = time.time()
# ----------------------------------------------------------------------------------------------
# en dataset
dataset = load_dataset(path='hosannaho/enzh', split="train")
dataset_train= dataset.select(range(10000))

def rename_column_en(example):
    example["instruction"] = "translate english to chinese."
    example["input"] = example.pop("english")
    example["output"] = example.pop("chinese")
    return example


renamed_dataset_en = dataset_train.map(rename_column_en, remove_columns=["english", "chinese"])
print(renamed_dataset_en['instruction'][1])
print(renamed_dataset_en['input'][1])
print(renamed_dataset_en['output'][1])
print(f"length of dataset: {len(dataset_train)}")


# for i in range(len(dataset_train)):
#     en = dataset_train['english'][i]
#     zh = dataset_train['chinese'][i]
#     train_dataset.append({'text': 'translate english to chinese:\n input:' + en + '\noutput:' + zh + EOS_TOKEN})

print("finish to load the english to chinese dataset\n")
# ----------------------------------------------------------------------------------------------
# ko dataset
dataset = load_dataset(path='traintogpb/aihub-kozh-translation-integrated-base-1m', split="train")
dataset_train = dataset.select(range(10000))

def rename_column_ko(example):
    example["instruction"] = "translate korean to chinese."
    example["input"] = example.pop("ko")
    example["output"] = example.pop("zh")
    return example

renamed_dataset_ko = dataset_train.map(rename_column_ko, remove_columns=["ko", "zh", 'source'])
print(renamed_dataset_ko['instruction'][1])
print(renamed_dataset_ko['input'][1])
print(renamed_dataset_ko['output'][1])
print(f"length of dataset: {len(dataset_train)}")

# for i in range(len(dataset_train)):
#     ko = dataset_train['ko'][i]
#     zh = dataset_train['zh'][i]
#     train_dataset.append({'text': 'translate korean to chinese:\n input:' + ko + '\noutput:' + zh + EOS_TOKEN})

print("finish to load the korean to chinese dataset\n")
# ----------------------------------------------------------------------------------------------
# ja dataset
dataset = load_dataset(path='larryvrh/WikiMatrix-v1-Ja_Zh-filtered', split="train")
dataset_train= dataset.select(range(10000))

def rename_column_ja(example):
    example["instruction"] = "translate japanese to chinese."
    example["input"] = example.pop("ja")
    example["output"] = example.pop("zh")
    return example

renamed_dataset_ja = dataset_train.map(rename_column_ja, remove_columns=["ja", "zh"])
print(renamed_dataset_ja['instruction'][1])
print(renamed_dataset_ja['input'][1])
print(renamed_dataset_ja['output'][1])

# for i in range(len(dataset_train)):
#     ja = dataset_train['ja'][i]
#     zh = dataset_train['zh'][i]
#     train_dataset.append({'text': 'translate japanese to chinese:\n input:' + ja + '\noutput:' + zh + EOS_TOKEN})

print("finish to load the japanese to chinese dataset\n")

# ----------------------------------------------------------------------------------------------
combined_dataset = concatenate_datasets([renamed_dataset_en, renamed_dataset_ko, renamed_dataset_ja])

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizers.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

combined_dataset = combined_dataset.map(formatting_prompts_func, batched = True,)

t1 = time.time()
t = t1 - t0
print(f"the total length of the dataset is: {len(combined_dataset)}, using time: {t}")
print(combined_dataset)

# ----------------------------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=token,
    torch_dtype=torch.float16,
    device_map=device
)
# ----------------------------------------------------------------------------------------------
# 配置lora
target_modules = ["q_proj", "v_proj",]
peft_config = LoraConfig(r=8,
                         lora_alpha=16,
                         target_modules=target_modules,
                         lora_dropout=0.05,
                         bias='none',
                         task_type='CAUSAL_LM')
# ----------------------------------------------------------------------------------------------
# 配置訓練參數
output_dir = 'C:\\Users\\Lily\\Desktop\\python\\llama\\mt'
torch.cuda.empty_cache()

training_aruments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    optim='adamw_torch',
    learning_rate=2e-5,
    dataset_text_field='text',
    save_steps=5000,
    logging_strategy= 'steps',
    logging_first_step= True,
    logging_steps=100,
    report_to="wandb",
    # eval_strategy='steps',
    max_steps=60000,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    #max_grad_norm=0.3,
    bf16=True,
    #lr_scheduler_type='cosine',
    #warmup_steps=600,
    max_seq_length=512,
)
# ----------------------------------------------------------------------------------------------
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.config.use_cache = False
# ----------------------------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=combined_dataset,
    peft_config=peft_config,
    tokenizer=tokenizers,
    args=training_aruments
)

file_in_folder = os.listdir(output_dir)
flag_check = True if file_in_folder else False
if flag_check:
    trainer.train(resume_from_checkpoint= flag_check)
else:
    trainer.train()
# ----------------------------------------------------------------------------------------------
#取訓練log
#log_history = trainer.state.log_history
#for log in log_history:
#    if "loss" in log:
#        print(f"Step {log['step']}: Loss = {log['loss']}")