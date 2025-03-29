import pandas as pd
import numpy as np 
import torch
from tqdm import tqdm
from random import randrange
import plotly.express as px
import gc
import os
import wandb
from kaggle_secrets import UserSecretsClient

from datasets import load_dataset,concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,
                          BitsAndBytesConfig,Seq2SeqTrainingArguments,Seq2SeqTrainer)
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training , TaskType
import evaluate

from accelerate import Accelerator
accelerator = Accelerator()
user_secrets = UserSecretsClient()

class cVariables: 
    
    __shared_instance = None
    @staticmethod
    def get_instance():
        if cVariables.__shared_instance == None: cVariables()
        return __shared_instance
    def __init__(self):
        if cVariables.__shared_instance != None : raise Exception("This class is a singleton class !")
        else:  cVariables.__shared_instance = self
        #----------------------
        self.ATTEMPT_NO = 0 # i reduce sample of data to able to train model because size in gpu
        # this parameter  is responsible for dividing data into and out
        # get_SizeSampleTrain and get_SizeSampleEval to return start and end of sample of data
        #----------------------

    def get_SizeSampleTrain(self):
        assert self.ATTEMPT_NO < 7 , "ATTEMPT_NO should be less than 7"
        TRAIN_SIZE=14732//6
        TRAIN_LIST = [i*TRAIN_SIZE for i in range(0,8)]
        return  TRAIN_LIST[self.ATTEMPT_NO] , TRAIN_LIST[self.ATTEMPT_NO+1]
    def get_SizeSampleEval(self):
        assert self.ATTEMPT_NO < 7 , "ATTEMPT_NO should be less than 7"
        if self.ATTEMPT_NO == 6 :
            return TRAIN_LIST[-1] , 14732
        EVAL_SIZE=818//6
        EVAL_LIST = [i*EVAL_SIZE for i in range(0,7)]
        return EVAL_LIST[self.ATTEMPT_NO] , EVAL_LIST[self.ATTEMPT_NO+1]

    Paths={
    'data' : 'samsum',
    'model': 'google/flan-t5-large',       
    'new_checkpoint': f'FlanT5Summarization-samsum',
    'wandb_proj': 'Summarization by Finetuning FlanT5-LoRA',
    'wandb_run':f'flant5Summarization',
    }
    Hayperparameters={
     'max_source_length':512,
     'max_target_length':128,
     'batch_size_train':128,
     'batch_size_eval':64,
     'epochs':3,
     'lr':3e-5,
     'l2':0.01,
    }
    Tokens={'huggingface' :user_secrets.get_secret("huggingface"),
            'wandb': user_secrets.get_secret("wandb")}
    
var = cVariables()

def clear_gpu():
    print(gc.collect()) 
    torch.cuda.empty_cache()
    print(gc.collect())

rouge = evaluate.load("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Ensure the predictions and labels are in the correct format
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    pred_ids = np.argmax(pred_ids, axis=-1) if pred_ids.ndim == 3 else pred_ids

    # Convert tensors to lists
    pred_ids = pred_ids.tolist()
    labels_ids = labels_ids.tolist()

    # Decode generated summaries and labels (converting token IDs back to text)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids = [[token for token in label if token != -100] for label in labels_ids]
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Compute ROUGE scores
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)

    return {
        "rouge1": rouge_output["rouge1"],
        "rouge2": rouge_output["rouge2"],
        "rougeL": rouge_output["rougeL"],
        "rougeLsum": rouge_output["rougeLsum"],
    }

tokenizer = AutoTokenizer.from_pretrained(var.Paths['model'],token=var.Tokens['huggingface'])

def process_dataset(data):
    inputs = ["summarize: " + item for item in data["dialogue"]]

    model_inputs = tokenizer(inputs,add_special_tokens=True,
                max_length=var.Hayperparameters['max_source_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt')
    model_target = tokenizer(inputs,add_special_tokens=True,
                max_length=var.Hayperparameters['max_target_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt')
    model_target["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in model_target] for model_target in model_target["input_ids"]]
    
    model_inputs["labels"] = model_target["input_ids"]
    return model_inputs

os.environ["WANDB_API_KEY"] = var.Tokens['wandb']
os.environ["WANDB_DEBUG"] = "true"
os.environ["WANDB_PROJECT"]=var.Paths['wandb_proj']
os.environ["WANDB_NAME"] = var.Paths['new_checkpoint']

wandb.init()

dataset = load_dataset(var.Paths['data'], trust_remote_code=True)

tokenized_dataset = dataset.map(process_dataset, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# Slice the mapped datasets to get the smaller samples
start_train, end_train = var.get_SizeSampleTrain()
start_eval, end_eval = var.get_SizeSampleEval()

train_dataset = tokenized_dataset['train'].select(range(start_train,end_train))
validation_dataset = tokenized_dataset['validation'].select(range(start_eval,end_eval))

quantization_config = BitsAndBytesConfig(load_in_4bit=True,)

model = AutoModelForSeq2SeqLM.from_pretrained(var.Paths['model'],
                                            quantization_config=quantization_config,
                                            device_map="auto",
                                            token=var.Tokens['huggingface'])
clear_gpu()

# Define LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],

    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# prepare int-8 model for training
model = prepare_model_for_kbit_training(model)

# # add LoRA adaptor
model = get_peft_model(model, lora_config)

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

training_args = Seq2SeqTrainingArguments(
    output_dir=var.Paths['new_checkpoint'],
    num_train_epochs=var.Hayperparameters['epochs'],
    
    evaluation_strategy = 'steps',
    save_strategy = 'steps',
    load_best_model_at_end = True,
    logging_steps =5,
    eval_steps = 5,
    save_total_limit =2,
    predict_with_generate=True , # For generating summaries during evaluation

    
    lr_scheduler_type = "cosine",
    learning_rate = var.Hayperparameters['lr'],
    optim="adamw_torch",
    
    auto_find_batch_size=True,
    per_device_train_batch_size = var.Hayperparameters['batch_size_train'],
    per_device_eval_batch_size = var.Hayperparameters['batch_size_eval'],
    weight_decay = var.Hayperparameters['l2'],
    warmup_ratio=0.1,
    gradient_accumulation_steps=4,
    
    push_to_hub=True,
    hub_private_repo=True,
    hub_token=var.Tokens['huggingface'],
    run_name=var.Paths['new_checkpoint'],

    report_to=['wandb'],
)
clear_gpu()

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

model.config.use_cache = False
clear_gpu()

trainer.train()
clear_gpu()