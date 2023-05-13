from pykeops.torch import LazyTensor
from transformers import Trainer, TrainerCallback

import torch

from tqdm import tqdm

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

from transformers import AutoTokenizer

import torch.nn.functional as F
from torch import nn

import re, json

import datetime
import umap

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

dim_reduc = umap.UMAP(random_state=10, metric='euclidean')

uscode_reg = r'(?<=\s)([\d]+[a-z]*)\s+U.S.C. §+ (([0-9]+[a-z]*[-–]*)+)(\(([a-z])\))*(\(([\d])\))*'
precedence_reg = r'[0-9]{3} .*?\([0-9]{4}\)'

date_reg = r'. ([A-Z]{1}[a-z]+ [0-9]{1,2}, [0-9]{4})'

def preprocess(to_extract):
    codes = []
    #parser.parse(to_extract[:500], fuzzy=True)
    date_str = re.search(date_reg, to_extract[:500])
    try:
        date = datetime.datetime.strptime(date_str.group(1), '%B %d, %Y')
    except Exception as e:
        date = None
    
    c = re.finditer(uscode_reg, to_extract)
    masked = to_extract
#     citations = get_citations(to_extract)
    if c:
        for code in c:
            token = 'USC_{0}'.format(code.group(1), code.group(2), code.group(5))
            
            if int(re.sub('[^0-9]', '', code.group(1))) < 100:
                #token = code
                codes.append(code.group(0))
                #masked = masked.replace(code.group(0), ' <mask>')
                to_extract = to_extract.replace(code.group(0), ' ')
                
            else:
                masked = masked.replace(code.group(0), ' ')
                to_extract = to_extract.replace(code.group(0), ' ')
    
    to_extract = re.sub(precedence_reg, ' ', to_extract)
    to_extract = re.sub(r'(\<\/*[a-z0-9 ="-]*\>)', ' ', to_extract)
    to_extract = re.sub(r'(\[[0-9]+\])', ' ', to_extract)
    to_extract = to_extract.replace('&amp', '&')
    to_extract = to_extract.replace(r'[\n\[\]]', ' ')
    to_extract = re.sub(r'\([A-Za-z0-9]+\)', ' ', to_extract)
    to_extract = to_extract.encode('ascii', errors='ignore').decode()
    to_extract = re.sub(r'[\s]{2,}', '', to_extract)
    
    return [to_extract, masked, codes, date, len(to_extract)]


def KMeans_cosine(x, model=None, K=5, Niter=10):
    """Implements Lloyd's algorithm for the Cosine similarity metric."""

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids
    # Normalize the centroids for the cosine similarity:
    c = torch.nn.functional.normalize(c, dim=1, p=2)

    x_i = LazyTensor(x.unsqueeze(1))  # (N, 1, D) samples
    c_j = LazyTensor(c.unsqueeze(0))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
        
        cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster
        
        clus = S_ij.max_argmax(dim=1) # distance to the nearest cluster
#         print(cl)
        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Normalize the centroids, in place:
        #c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

    print(
        f"K-means for the cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:"
    )

    return clus, c




def CustomMetrics(ep):
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
            'macro-r': recall_score(ep.label_ids, pred_round, average='macro'),
            'macro-p': precision_score(ep.label_ids, pred_round, average='macro'),
            'micro-f1': f1_score(ep.label_ids, pred_round, average='micro'),
            'hamming': hamming_loss(ep.label_ids, pred_round)
           }

def PrototypeMetrics(ep):
    #custom softmax
    print(ep.predictions[0])
    if len(ep.predictions[0][1]) > 1:
        m = np.max(ep.predictions[0][1],axis=1,keepdims=True) #returns max of each row and keeps same dims
    else:
        m = np.max(ep.predictions[0][1])
    e_x = np.exp(ep.predictions[0][1] - m) #subtracts each row with its max value
    s = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / s 
    
    pred_round = np.round(f_x).astype(np.float16)
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
            'macro-f1': f1_score(ep.label_ids.astype(np.float16), pred_round, average='macro'),
            'macro-r': recall_score(ep.label_ids.astype(np.float16), pred_round, average='macro'),
            'macro-p': precision_score(ep.label_ids.astype(np.float16), pred_round, average='macro'),
            'micro-f1': f1_score(ep.label_ids.astype(np.float16), pred_round, average='micro'),
            'hamming': hamming_loss(ep.label_ids.astype(np.float16), pred_round)
           }


def plot_umap(normalized_df, colour, prototypes, name, definitions=None):
    standard_embedding = dim_reduc.fit_transform(normalized_df)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=colour, cmap='YlGnBu', s=1)
    
    prot_embedding = dim_reduc.transform(prototypes)
    
    plt.scatter(prot_embedding[:, 0], prot_embedding[:, 1], color='r', s=8)
    
    if definitions is not None:
        def_embedding = dim_reduc.transform(definitions)
        plt.scatter(def_embedding[:, 0], def_embedding[:, 1], color='m', s=8)
    
    plt.savefig(name, dpi=200, bb_inches='tight')
    plt.show()

def plot_tsne(normalized_df, colour):
    standard_embedding = TSNE(n_components=2, perplexity=8, learning_rate='auto',init='random').fit_transform(normalized_df)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=colour, s=1, cmap='bwr')


def load_defs(model_ckpt, l):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    with open('citation_map.json', 'r') as f:
        label_dict = json.loads(f.readline())
    citations = list(label_dict.items())
    nonproced = { '11_523_(a)': 1, '42_2000e_': 7, '11_101_': 8, '42_405_(g)': 10,  '29_1132_(a)': 17, '11_362_(a)': 18, '21_841_(a)': 21, '18_1961_': 22, '29_1002_': 24, '11_727_(a)': 25, '18_922_(g)': 26,  '11_362_': 29,  '15_1_': 31, '11_541_(a)': 32, '15_1125_(a)': 33, '29_1001_': 34, '35_112_': 37, '15_78j_(b)': 40, '18_924_(c)': 41,  '11_547_(b)': 49, '42_2000e-2_(a)': 50, '11_362_(d)': 52, '11_522_(f)': 53, '18_371_': 54, '18_1341_': 55, '5_552_(a)': 56, '42_9601_': 57, '42_423_(d)': 58, '8_1101_(a)': 60, '29_621_': 61, '42_1396a_(a)': 63, '18_2_': 66, '11_522_(b)': 67, '15_78u-4_(b)': 68, '42_9607_(a)': 69, '21_841_(b)': 73, '18_1962_(c)': 76,  '11_522_(d)': 80, '15_1114_': 81, '21_846_': 85, '42_12101_': 86, '11_541_': 90, '5_702_': 91,  '29_1144_(a)': 93, '42_12102_': 94}


    defs = []

    if l == 45: # non-precedential
        for k, v in nonproced.items():
            with open('definitions/{0}-{1}.json'.format(v, k), 'r') as f:
                defs.append(f.readline())
    else:
        for i in range(l):
            with open('definitions/{0}-{1}.json'.format(citations[i][1], citations[i][0]), 'r') as f:
                defs.append(f.readline())

    defs_cleaned = []
    for d in defs:
        to_extract = preprocess(d)[0]

        to_extract = tokenizer(to_extract, max_length=(4096 if model_ckpt == 'allenai/longformer-base-4096' else 512), truncation=True, padding='max_length', return_tensors='pt')['input_ids']
        
        #print(to_extract.shape)
        defs_cleaned.append(to_extract)
    return defs_cleaned