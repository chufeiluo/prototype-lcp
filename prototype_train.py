import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import transformers
import datasets

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer)

#from eyecite import get_citations
import numpy as np
import re
import pandas as pd
from tqdm.notebook import tqdm


from transformers import AutoModel,AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torchinfo import summary

from scripts.prototype_model import PrototypeModel, PrototypeTrainer, PrototypeCallback


from scripts.utils import *

import json
import math, argparse

parser = argparse.ArgumentParser(description='Help me god')
parser.add_argument('-l', '--label-len', help='label length', type=int, default=45)
parser.add_argument('-f', '--form', help='format', default='red2')
parser.add_argument('-c', '--checkpoint', help='base checkpoint', default='')
parser.add_argument('-r', '--resume', help='resume from checkpoint', action='store_true')

parser.add_argument('-m', '--model', help='name of the model', default='nlpaueb/legal-bert-base-uncased')

parser.add_argument('-d', '--definitions', help='whether or not to use definitions to build prototypes', action='store_true')

parser.add_argument('-n', '--num_prototypes', help='number of discovered prototypes per class', type=int, default=5)
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=20)
parser.add_argument('-b', '--batch_size', help='batch size for training', type=int, default=8)
args = parser.parse_args()


form = args.form
checkpoint = args.checkpoint
l = args.label_len
res = args.resume
model = args.model
d = args.definitions

epochs = args.epoch
batch_size= args.batch_size
num_prototypes = args.num_prototypes

# form = 'red2' # 512 for no context, red2 for +/-2 sentence context, red4, etc. - this doesn't do the preprocessing, you have to run preprocessing.py
# l = 45 # number of target citations
# checkpoint = 66000 # the best baseline checkpoint from finetune.py

# resume = True

device = 'cuda'

if checkpoint == '':
    model_ckpt = model
else:
    model_ckpt = f'baseline/baseline-{form}-{l}-{model}/checkpoint-{checkpoint}'

tokenizer = AutoTokenizer.from_pretrained(model)


# !mkdir -p f'prototypes-{form}-{l}'


def tokenize_and_encode_in(examples):
    return tokenizer(examples['input'], max_length=512, truncation=True, padding=True)

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




train = load_dataset('json', data_files=f'data/{form}/train-citationcleaned-{l}.json', split='train')
val = load_dataset('json', data_files=f'data/{form}/val-citationcleaned-{l}.json', split='train')
test = load_dataset('json', data_files=f'data/{form}/test-citationcleaned-{l}.json', split='train')

train_enc = prep(train, 'train', form, l)
val_enc = prep(val, 'val', form, l)
test_enc = prep(test, 'test', form, l)


model = AutoModel.from_pretrained(model_ckpt).to(device)
if d:
    defs_cleaned = load_defs(model_ckpt, l)
    defs = torch.stack(defs_cleaned).squeeze().to(device)
else:
    defs = None
    
model = PrototypeModel(model,config=transformers.AutoConfig.from_pretrained(model_ckpt), N_input=768, N_output=l, definitions=defs, num_prototypes_per_class=num_prototypes)

#out = model(sample, )
print(summary(model))

#model = RobertaBaseline(roberta, N_input=512*768, N_output=20)
model.roberta.resize_token_embeddings(len(tokenizer))
#multitask_model.encoder.resize_token_embeddings(len(tokenizer))


optimizer = torch.optim.AdamW(model.parameters(),
                lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
            )
    # Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 500, # Default value in run_glue.py
                                            num_training_steps = math.ceil((len(train_enc)/batch_size)*epochs))


batch_size = 8
run_name = f'prototype-{form}-{l}-{checkpoint}-defs'

args = TrainingArguments(
    output_dir=run_name,
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
    save_total_limit = 5, 
    metric_for_best_model='macro-f1'# Only last 5 models are saved. Older ones are deleted.
)
#del trainer
trainer = PrototypeTrainer(model=model, 
                  args=args, 
                  train_dataset=train_enc, 
                  eval_dataset=val_enc, 
                  tokenizer=tokenizer, 
                  compute_metrics=PrototypeMetrics,
                  callbacks=[PrototypeCallback()]
                 )


trainer.train(resume_from_checkpoint=res)