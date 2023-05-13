import os

import transformers
import datasets

import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer,
                            AutoModelForMaskedLM,AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup)
import numpy as np
import re
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, hamming_loss


import argparse
import math

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels = labels.unsqueeze(0)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss
        
        


def customMetrics(ep):
    print(ep.predictions)
    print(ep.predictions.shape)
    m = np.max(ep.predictions,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(ep.predictions - m) #subtracts each row with its max value
    s = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / s 
    
    pred_round = np.round(f_x)
    print(pred_round, ep.label_ids)
    pred = pred_round == ep.label_ids[:len(pred_round)]
    
    negatives = np.logical_not(ep.label_ids[:len(pred)])
    guess = np.logical_and(pred, ep.label_ids[:len(pred)])
    not_cite = np.logical_and(pred, negatives)
    #print(np.sum(pred), np.sum(ep.label_ids[:len(pred)]), guess, not_cite)
    return {'tp': int(np.sum(guess.astype(float))),
           'fn': (int(np.sum(ep.label_ids[:len(pred)])) - int(np.sum(guess.astype(float)))),
            'tn': int(np.sum(not_cite.astype(float))),
           'fp': (int(np.sum(negatives)) - int(np.sum(not_cite.astype(float)))),
            'macro-f1': f1_score(ep.label_ids, pred_round, average='macro'),
            'micro-f1': f1_score(ep.label_ids, pred_round, average='micro'),
            'hamming': hamming_loss(ep.label_ids, pred_round)
           }



parser = argparse.ArgumentParser(description='Help me god')
parser.add_argument('-l', '--label-len', help='label length')
parser.add_argument('-f', '--form', help='format')
parser.add_argument('-r', '--resume', help='resume from checkpoint', action='store_true')

parser.add_argument('-m', '--model', help='model to fine-tune')
parser.add_argument('-t', '--token-len', default='512', help='tokenizer max length')
args = parser.parse_args()

form = args.form
l = int(args.label_len)
resume = args.resume
max_len = int(args.token_len)

print('performing fine-tuning with model {2} label length {0} and format {1}'.format(l, form, args.model))
print('resuming from checkpoint={0}'.format(args.resume))
print('loading data')
# load data


model_ckpt = args.model

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=l).to('cuda')

device = 'cuda'

def prep(data, name, preproc, label_len):
    cols = data.column_names
    cols.remove("labels")
    print(cols)
    data_enc = data.map(tokenize_and_encode_in, batched=True, remove_columns=cols)
    
    data_enc.set_format("torch")
    data_enc = (data_enc
              .map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
              .rename_column("float_labels", "labels"))
    
    return data_enc

def tokenize_and_encode_in(examples):
    return tokenizer(examples['input'], max_length=max_len, truncation=True, padding=True)

train = load_dataset('json', data_files=f'data/{form}/train-citationcleaned-{l}.json', split='train')
val = load_dataset('json', data_files=f'data/{form}/val-citationcleaned-{l}.json', split='train')
test = load_dataset('json', data_files=f'data/{form}/test-citationcleaned-{l}.json', split='train')


train_enc = prep(train, 'train', form, l)
val_enc = prep(val, 'val', form, l)
test_enc = prep(test, 'test', form, l)


#model = RobertaBaseline(roberta, N_input=512*768, N_output=20)
model.resize_token_embeddings(len(tokenizer))
#multitask_model.encoder.resize_token_embeddings(len(tokenizer))

epochs = 20
batch_size=4

run_name = 'baseline-{0}-{1}-{2}'.format(form, l, model_ckpt)

args = TrainingArguments(
    output_dir='baseline/{0}'.format(run_name),
    evaluation_strategy = "steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    do_eval=True,
    logging_first_step=True,
    logging_steps=500,
    dataloader_num_workers=16,
    do_train=True,
    fp16=True,
    run_name = run_name,
    report_to = 'wandb',
    eval_accumulation_steps=100,
    save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
)
#del trainer
trainer = CustomTrainer(model=model, 
                  args=args, 
                  train_dataset=train_enc, 
                  eval_dataset=val_enc, 
                  tokenizer=tokenizer, 
                  compute_metrics=customMetrics
                 )

trainer.train(resume_from_checkpoint=resume)

print(trainer.predict(test_enc))
