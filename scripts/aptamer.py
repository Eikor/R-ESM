# Script for preparing snRNA data
import os
os.sys.path.append('/junde/R-ESM')
import dist_misc
from tqdm import tqdm
from data import RNADataset, Alphabet_RNA, MaskedBatchConverter
import pandas as pd
from RESM import RESM
import torch
import torch.nn as nn
from torch.utils.data import DistributedSampler, Dataset
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.utils.fixes import loguniform
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def extract_embedding(aptamer_url, pretrain_url, repr_layers=[12], save_name=None):
    seq_data = pd.read_csv(aptamer_url,sep=',', header=0)
    # load training result
    pretrain_info = torch.load(pretrain_url)
    args = pretrain_info['args']
    alphabet = Alphabet_RNA.RNA(coden_size=args.coden_size)
    model = RESM(alphabet, num_layers=args.num_layers, embed_dim=args.embed_dim, attention_heads=20).cuda()
    model.load_state_dict(pretrain_info['model'])
    model.eval()
    # construct data loader
    class aptamer_dataset(Dataset):
        def __init__(self, seq_data) -> None:
            super().__init__()
            self.seqs = seq_data
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, index):
            return ('{:0>5d}'.format(index), self.seqs.loc[index, 'seq'])
        
    data_loader = torch.utils.data.DataLoader(
        aptamer_dataset(seq_data), collate_fn=MaskedBatchConverter(alphabet=alphabet, truncation_seq_length=99999), num_workers=16)

    labels = []
    features = []
    seqid = []
    for batch in tqdm(data_loader):
        (label, strs, toks, masktoks, masks) = batch
        
        # if label[0].split()[0] != 'URS0000E7F388':
        #     continue
        seqid.append(strs)
        labels.append(-1)

        toks = toks.to(device="cuda", non_blocking=True) 
        with torch.no_grad():
            out = model(toks, repr_layers=repr_layers)
        features.append(out["representations"][repr_layers[-1]][0].mean(0))

    features = [feature.tolist() for feature in features]
    dataset = pd.DataFrame(list(zip(seqid, features, labels)), columns=['seqid', 'features', 'labels'])
    dataset.index = dataset['seqid']
    if save_name is not None:
        dataset.to_parquet(save_name)
    return dataset

def load_labels_to_dataset(dataset, labels_file):
    labels_file.index = labels_file.seqid
    for seqid, info in tqdm(labels_file.iterrows()):
        try: 
            dataset.loc[info.seqid, 'labels'] = info.labels
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


if __name__ == '__main__':
    aptamer_url = '/junde/aptamer.txt'
    pretrain_url = 'snRNA_35M_coden1_100epoch/checkpoint-99.pth'
    feature_url = 'snRNA_35M_coden1_100epoch/35M_aptamer_feature_cache.pd.parquet'
    exp_name = 'snRNA_35M_coden1_100epoch/35M_coden1_aptamer'
    labels_file = pd.read_csv('/junde/aptamer_kd.txt',sep=',', header=0, names=['seqid', 'labels'])

    seq_data = pd.read_csv(aptamer_url,sep=',', header=0)
    # load training result
    pretrain_info = torch.load(pretrain_url)
    args = pretrain_info['args']
    alphabet = Alphabet_RNA.RNA(coden_size=args.coden_size)
    model = RESM(alphabet, num_layers=args.num_layers, embed_dim=args.embed_dim, attention_heads=20).cuda()
    model.load_state_dict(pretrain_info['model'])
    model.eval()

    # construct data loader
    class aptamer_dataset(Dataset):
        def __init__(self, seq_data) -> None:
            super().__init__()
            self.seqs = seq_data
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, index):
            return (self.seqs.loc[index, 'labels'], self.seqs.loc[index, 'seqid'])
        
    data_loader = torch.utils.data.DataLoader(
        aptamer_dataset(labels_file), collate_fn=MaskedBatchConverter(alphabet=alphabet, truncation_seq_length=99999), num_workers=16)

    labels = []
    features = []
    seqid = []
    for batch in tqdm(data_loader):
        (label, strs, toks, masktoks, masks) = batch
        
        # if label[0].split()[0] != 'URS0000E7F388':
        #     continue
        seqid.append(strs[0])
        labels.append(label)

        toks = toks.to(device="cuda", non_blocking=True) 
        with torch.no_grad():
            out = model(toks, repr_layers=[12])
        features.append(out["representations"][12][0].mean(0))

    features = [feature.tolist() for feature in features]
    dataset = pd.DataFrame(list(zip(seqid, features, labels)), columns=['seqid', 'features', 'labels'])
    dataset.index = dataset['seqid']

    # load_labels_to_dataset(dataset, labels_file)

    trainset = dataset.sample(30, random_state=0)
    testset = dataset.drop(trainset.index)

    X, y = np.array(trainset.features.to_list()), np.array(trainset.labels.to_list())

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e5,
            random_state=0).fit(X, y)
    print(gpr.score(X, y))

    X, y = np.array(testset.features.to_list()), np.array(testset.labels.to_list())
    print(gpr.predict(X, return_std=True)[0])



        


