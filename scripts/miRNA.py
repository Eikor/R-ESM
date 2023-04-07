import os
os.sys.path.append('/junde/R-ESM')
from tqdm import tqdm
from data import RNADataset, Alphabet_RNA, MaskedBatchConverter, DistributedBatchSampler
import pandas as pd
from RESM import RESM
import torch
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.utils.fixes import loguniform
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from finetune import finetune_cls

def extract_miRNA(fasta_name):
    fasta_name = '/junde/rnacentral_active_cleaned.fasta'

    new_file = open('/junde/snRNA.fasta','a')

    with open(fasta_name, "r") as infile:
        for line_idx, line in tqdm(enumerate(infile)):
            if line.startswith(">"):
                flag = False
                if 'snRNA' in line:
                    new_file.write(line)
                    flag=True
                    continue
            if flag:
                new_file.write(line)

if __name__ == '__main__':
    extract_miRNA()