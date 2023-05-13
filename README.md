# Prototype-based Interpretability for Legal Citation Prediction

Accepted to Findings of ACL 2023. If you use this code, please cite our paper:
```
@proceedings{luo2023prototype,
    title = "Prototype-Based Interpretability for Legal Citation Prediction",
    editor = "Luo, Chu Fei and Bhambhoria, Rohan and Dahan, Samuel and Zhu, Xiaodan",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2305.16490",
}

```

## Requirements
We recommend installing conda and [building a virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file) from environment.yml
Then, run `pip install -r requirements.txt`

## How to run
1. Obtain the raw data
    1. If you would like the official, preprocessed data in its train/val/test split from the paper, please email me at 14cfl@queensu.ca
    2. Otherwise, refer to **Preparing the data from scratch** below
2. To prepare the text, including generating the surrounding contexts, run:
```
preprocessing.py --label-len {100,20,45}
```
Please refer to **Experiment details** for explanations of label-len (number of input labels). The best performing model uses 45 labels. 
2. To fine-tune legalBERT, run:
```
train.py --label-len {100,20,45} --form {512,red4,red2} --model nlpaueb/legal-bert-base-uncased
```
Please refer to **Experiment details** for explanations on form (input context) and label-len (number of input labels)
3. The prototype-based training code is hosted in jupyter notebooks.
    - To train the legalBERT checkpoint with **precedent-based prototypes only**, use `prototype_train.py`
    - To train legalBERT with **precedent- and provision-based prototypes**, use the notebook `prototype_train.py -d`
    - To generate figures of the embedding space, run `latentspace-plot.ipynb`

Try `prototype_train.py --help` for more training options.

### Preparing the data from scratch
1. Bulk download opinion data from CourtListener: https://www.courtlistener.com/help/api/bulk-data/ and save it in the subdirectory `data/opinions`
2. Scrape US code definitions (we used LLI) and store them in separate text files with the format `lii/text/_uscode_text_{title}_{heading}{paragraph}.txt` (contact us if you would like a preprocessed version)
3. Run all the cells in `process citations.ipynb`


## Experiment details
- We allow variance in two aspects of the input:
    - **Number of input labels** to mitigate the long-tail problem. We take the top n by citation frequency, except for 45 labels which is filtered by experts.
    - **Input context** to compensate for the context length of language models, which is significantly smaller than court cases. Knowing the ground truth locations of the citations, we take m sentences before and after the target before removing the citation with regex.
- We construct prototypes from two data sources:
    - **Precedent-based,** i.e. clustering the training data
    - **Provision-based,** i.e. using the target citations' source text, which is provisions of legislation in our case