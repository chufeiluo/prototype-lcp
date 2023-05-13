import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import transformers
import datasets


import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer)

#from eyecite import get_citations
import numpy as np
import re
import pandas as pd
from tqdm import *
from datasets import Dataset

from nltk.tokenize import sent_tokenize

import datetime

import argparse
parser = argparse.ArgumentParser(description='Help me god')
parser.add_argument('-l', '--label-length', help='label length')
args = parser.parse_args()

from autocorrect import Speller
spell = Speller(lang='en')
n = 0 # initialization


batch_size=10
model_ckpt = 'allenai/longformer-base-4096'

max_len = (4096 if model_ckpt == 'allenai/longformer-base-4096' else 512)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

uscode_reg = r'(?<=\s)([\d]+[a-z]*)\s+U.S.C. §+ (([0-9]+[a-z]*[-–]*)+)(\(([a-z])\))*(\(([\d])\))*'
precedence_reg = r'[0-9]{3} .*?\([0-9]{4}\)'

date_reg = r'. ([A-Z]{1}[a-z]+ [0-9]{1,2}, [0-9]{4})'

def tokenize(data):
    global n
    text = []
    sents = sent_tokenize(data)
    to_keep = set()
    for i in range(len(sents)):
        if '<mask>' in sents[i] or re.search('U.S.C.', sents[i]) is not None:
            to_keep.update(range(max(0,i-n), min(len(sents), i+n)))
    
    for k in to_keep:
        text.append(sents[k])

    return ' '.join(text)

def preprocess(to_extract):
    codes = []
    #parser.parse(to_extract[:500], fuzzy=True)
    date_str = re.search(date_reg, to_extract[:500])
    try:
        date = datetime.datetime.strptime(spell(date_str.group(1)), '%B %d, %Y')
    except Exception as e:
        date = None
    
    c = re.finditer(uscode_reg, to_extract)
    masked = to_extract
#     citations = get_citations(to_extract)

    global preproc
    if preproc.startswith('red'):
        to_extract = tokenize(to_extract)
    elif preproc == 'none':
        to_extract = to_extract[100:]
    if c:
        for code in c:
            token = 'USC_{0}'.format(code.group(1), code.group(2), code.group(5))
            
            if int(re.sub(r'[^0-9]', '', code.group(1))) < 100:
                #token = code
                codes.append(code.group(0))
                #masked = masked.replace(code.group(0), ' <mask>')
                to_extract = to_extract.replace(code.group(0), '')
                
            else:
                masked = masked.replace(code.group(0), '')
                to_extract = to_extract.replace(code.group(0), '')
    
    to_extract = re.sub(precedence_reg, ' ', to_extract)
    to_extract = re.sub(r'(\<\/*[a-z0-9 ="-]*\>)', ' ', to_extract)
    to_extract = re.sub(r'(\[[0-9]+\])', ' ', to_extract)
    to_extract = to_extract.replace('&amp', '&')
    to_extract = to_extract.replace(r'[\[\]]', ' ')
    to_extract = re.sub(r'\([0-9]+\)', ' ', to_extract)
    to_extract = re.sub(r'\[[a-zA-Z]\]', ' ', to_extract)
    to_extract = to_extract.encode('ascii', errors='ignore').decode()
    to_extract = re.sub(r'[\s]{2,}', ' ', to_extract)
    
    return [to_extract, masked, codes, date, len(to_extract)]
    
    #return [to_extract, masked]
    
def tok(d):
    mlm = []
    p = [[], []]
    codes = []
    labels = []
    files = []
    doc_len = []
    
    dates = []

    for i in range(len(d['text'])):
        #print(text)
        data = preprocess(d['text'][i])
        p[0].append(data[0])
        p[1].append(data[1])
    #print(preproc[2])
        codes.append(data[2])
        doc_len.append(data[4])
        labels.append(d['labels'][i])
        files.append(d['files'][i])
    d = {'input': p[0], 'original': p[1], 'codes': codes, 'labels': labels, 'files': files}
    
    #print(len(p[0]), len(p[1]))
    
    
    #tokenizer.sanitize_special_tokens()
    #text = tokenizer(p[0], max_length=512, truncation=True, padding=True, return_tensors='pt')
    #mlm = tokenizer(p[1], max_length=512, truncation=True, padding=True, return_tensors='pt')
    
    return d


def tokenize_and_encode_in(examples):
    return tokenizer(examples['input'], max_length=max_len, truncation=True, padding=True)
def tokenize_and_encode_mlm(examples):
    return {'mlm_enc': tokenizer(examples['mlm'], max_length=max_len, truncation=True, padding=True)['input_ids']}


def load_data(name, text, label, label_len):
    #text = pd.read_json(f'./data/{name}_data_100.json')
    #label = pd.read_json(f'./data/{name}_label_100.json')

    label = [x[:label_len] for x in list(label.T.values)]
    
    if name == 'train':
        text['labels'] = label
#         text['test'] = text['labels'].apply(lambda s: s.sum())
#         text = text[text['test'] > 0].drop(columns=['test'])
        
        data = Dataset.from_pandas(text, split="train")
        data = data.remove_columns('__index_level_0__')
        
    else:
        if name == 'test':
            text = text.T
        data = Dataset.from_pandas(text, split="train")
        data = data.remove_columns('__index_level_0__')
        data = data.add_column(name='labels', column=label)
    
    return data

def prep(data, name, preproc, label_len):
    # path to train.csv test.csv and test_labels.csv
    
    
    data = data.map(tok, batched=True, remove_columns=['files', 'text', 'labels'])
    
    data.to_json(f'data/{preproc}/{name}-citationcleaned-{label_len}.json')
#    cols = data.column_names
#    cols.remove("labels")
#    print(cols)
#    data_enc = data.map(tokenize_and_encode_in, batched=True, remove_columns=cols)
    
#    data_enc.set_format("torch")
#    data_enc = (data_enc
#              .map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
#              .rename_column("float_labels", "labels"))
#    data_enc.to_json(f'data/{preproc}/{name}enc-{label_len}.json')


################# change these files if you want to run with your own data - expects a dataset with column names files, text, labels #########
print('loading training data')
train_text = pd.read_json(f'./data/train_data_100.json')
train_label = pd.read_json(f'./data/train_label_100.json')
print('loading validation data')
val_text = pd.read_json(f'./data/val_data_100.json')
val_label = pd.read_json(f'./data/val_label_100.json')
print('loading test data')
test_text = pd.read_json(f'./data/test_data_100.json')
test_label = pd.read_json(f'./data/test_label_100.json')
#######################

##### main code loop
formats = ['none', 'red2', 'red4', 'red10']

num_labels = [5, 20, 100]
n_vals = [0, 2, 4, 10]

for l in num_labels:

    print('preprocessing for labels of length {0}'.format(l))
    train = load_data('train',train_text, train_label, l)
    val = load_data('val', val_text, val_label, l)
    test = load_data('test', test_text, test_label, l)

    for i in range(len(formats)):
        print('preprocessing format: {0}'.format(formats[i]))
        n = n_vals[i]
        preproc = formats[i]
        prep(train, 'train', formats[i], l)
        prep(val, 'val', formats[i], l)
        prep(test, 'test', formats[i], l)
