# Script for preparing snRNA data
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

def read_labels(filename):
    fail_labels = []
    with open(filename,'r') as ff:
        for line in ff:
            fail_labels.append(line.strip())
    return fail_labels

def extract_snRNA(fasta_name):
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

def extract_embedding(snRNA_url, pretrain_url, repr_layers=[12], save_name=None):
    seq_data = RNADataset.from_file(snRNA_url)  
    # load training result
    pretrain_info = torch.load(pretrain_url)
    args = pretrain_info['args']
    alphabet = Alphabet_RNA.RNA(coden_size=args.coden_size)
    model = RESM(alphabet, num_layers=args.num_layers, embed_dim=args.embed_dim, attention_heads=20).cuda()
    model.load_state_dict(pretrain_info['model'])
    model.eval()
    # construct data loader
    data_loader = torch.utils.data.DataLoader(
        seq_data, collate_fn=MaskedBatchConverter(alphabet=alphabet, truncation_seq_length=99999), num_workers=16)

    labels = []
    features = []
    seqid = []
    seqs = []
    for batch in tqdm(data_loader):
        (label, strs, toks, masktoks, masks) = batch
        
        # if label[0].split()[0] != 'URS0000E7F388':
        #     continue
        seqid.append(label[0].split()[0])
        labels.append(-1)
        seqs.append(strs)
        toks = toks.to(device="cuda", non_blocking=True) 
        with torch.no_grad():
            out = model(toks, repr_layers=repr_layers)
        features.append(out["representations"][repr_layers[-1]][0].mean(0))

    features = [feature.tolist() for feature in features]
    seqs = [seq.tolist() for seq in seqs]
    dataset = pd.DataFrame(list(zip(seqid, seqs,features, labels)), columns=['seqid', 'seq', 'features', 'labels'])

    if save_name is not None:
        dataset.to_parquet(save_name)
    return dataset

def load_labels_to_dataset(dataset, labels_file, min_count=1000):
    dataset.index = dataset['seqid']
    label_count = labels_file.labels.value_counts()
    valid_labels = []
    for name, count in zip(label_count.index, label_count.values):
        if count > min_count and name != 'NaN':
            valid_labels.append(name)
    label_to_idx = {label: i for i, label in enumerate(valid_labels)}

    for seqid, info in tqdm(labels_file.iterrows()):
        try: 
            label_idx = label_to_idx[info.labels]
        except: # invalid labels
            label_idx = -1
        try: 
            dataset.loc[seqid, 'labels'] = label_idx
        except: # invalid sequences
            print(seqid) # never execute
            pass

    dataset = dataset.dropna() # drop invalid sequence     
    return dataset, valid_labels

def run_svm(dataset, label_names, save_name=None):
    # split for training
    trainset = dataset.loc[dataset.labels != -1].groupby('labels').sample(500, random_state=0)
    
    X_train, y_train = trainset.features, trainset.labels

    X_train = np.array(X_train.tolist())
    y_train = np.array(y_train.tolist())

    # training
    param_grid = {
        "C": loguniform(1, 1e5),
        # "gamma": loguniform(1e-3, 1),
    }
    # cls = 
    cls = RandomizedSearchCV(
        svm.SVC(gamma='auto'), param_distributions=param_grid, 
        n_jobs=16, refit=False, random_state=0, n_iter=16, verbose=3)
    cls.fit(X_train, y_train)
    
    best_param = svm.SVC(kernel='rbf', C=cls.best_params_['C'], gamma='auto')
    # final testing avg from 3 trials
    for i in [111, 222, 333]:
        trainset = dataset.loc[dataset.labels != -1].groupby('labels').sample(500, random_state=i)
        testset = dataset.loc[dataset.labels != -1].drop(trainset.index)
        # best param
        X_train = np.array(X_train.tolist())
        y_train = np.array(y_train.tolist())
        
        best_param.fit(X_train, y_train)

        X_test, y_test = testset.features, testset.labels
        X_test = np.array([np.array(x) for x in X_test])
        y_test = np.array([np.array(y) for y in y_test])

        predictions = best_param.predict(X_test)
        cls_res = classification_report(y_test, predictions, target_names=label_names, digits=6)
        print(cls_res)
        if save_name is not None:
            with open(save_name + '_svm.txt', 'a') as f:
                f.write(cls_res)

            cm = confusion_matrix(y_test, predictions, labels=best_param.classes_)
            plt.figure(figsize=(10, 10))
            ax = sn.heatmap(cm, vmax=10000, annot=True, xticklabels=label_names, yticklabels=label_names)
            plt.savefig(save_name + f'_svm_{i}.png')

def tsne(dataset):
    X = np.array(dataset.features.tolist())
    # X = 
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random', perplexity=3).fit_transform(X)
    # plt.figure(figsize=)

if __name__ == '__main__':
    snRNA_url = '/junde/snRNA.fasta'
    pretrain_url = 'snRNA_35M_100epoch/checkpoint-99.pth'
    feature_url = 'snRNA_35M_100epoch/35M_snRNA_feature_seq_cache.pd.parquet'
    exp_name = 'snRNA_35M_100epoch/35M_snRNA_phylum'
    labels_file = pd.read_csv('snRNA_taxid_phylum.csv', index_col=0, header=0, names=['seqid', 'phylum_taxid', 'labels'])
    # labels_file = pd.read_csv('snRNA_Rfam_mapping.txt', sep='\t', header=0, names=['seqid', 'labels'], index_col=0)

    if os.path.exists(feature_url):
        dataset = pd.read_parquet(feature_url)
    else:
        dataset = extract_embedding(snRNA_url, pretrain_url, repr_layers=[12], save_name=feature_url) # extract embeddings

    # prepare labels         
    # pprint(labels_file.label.value_counts())
    dataset, label_names = load_labels_to_dataset(dataset, labels_file=labels_file)

    # SVM
    # run_svm(dataset, label_names, save_name=exp_name)

    # finetune
    for i in dataset.iterrows():
        dataset.loc[i[0], 'seq'] = dataset.loc[i[0], 'seq'].tolist()
    dataset = dataset.drop('seqid', axis=1).reset_index()
    trainset = dataset.loc[dataset.labels != -1].groupby('labels').sample(500, random_state=0)
    testset = dataset.loc[dataset.labels != -1].drop(trainset.index)
    finetune_cls(pretrain_url, trainset, label_names, exp_name)
    

        


