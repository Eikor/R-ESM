# Script for preparing snRNA data
import sys
import os
os.sys.path.append('/junde/R-ESM')
import dist_misc
from pprint import pprint
from tqdm import tqdm
from data import RNADataset, Alphabet_RNA, MaskedBatchConverter, DistributedBatchSampler
import pandas as pd
from RESM import RESM
import torch
import seaborn as sn
import matplotlib.pyplot as plt

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

def extract_embedding(data_loader, model, repr_layers=[12], save_name=None):
    phylums = []
    features = []
    seqid = []
    for batch in tqdm(data_loader):
        (labels, strs, toks, masktoks, masks) = batch
        
        seqid.append(labels[0].split()[0])
        try:
            phylums.append(labels_file.loc[seqid[-1]].phylum_name)
        except:
            seqid.pop(-1)
            continue

        if phylums[-1] not in valid_labels:
            seqid.pop(-1)
            phylums.pop(-1)
            continue

        toks = toks.to(device="cuda", non_blocking=True) 
        with torch.no_grad():
            out = model(toks, repr_layers=repr_layers)
        features.append(out["representations"][repr_layers[-1]][0].mean(0))

    features = [feature.tolist() for feature in features]
    phylums = [label_to_idx[phylum] for phylum in phylums]
    dataset = pd.DataFrame(list(zip(seqid, features, phylums)), columns=['seqid', 'features', 'phylum'])
    if save_name is not None:
        dataset.to_parquet(save_name)
    return dataset

def svm(trainset, testset):
    from sklearn import svm
    from sklearn.utils.fixes import loguniform
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np

    X_train, y_train = trainset.features, trainset.phylum
    X_train = np.array([np.array(x) for x in X_train])
    y_train = np.array([np.array(y) for y in y_train])

    param_grid = {
        "C": loguniform(1e3, 1e5),
        "gamma": loguniform(1e-4, 1e-1),
    }
    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)

    X_test, y_test = testset.features, testset.phylum
    X_test = np.array([np.array(x) for x in X_test])
    y_test = np.array([np.array(y) for y in y_test])

    predictions = rbf.predict(X_test)
    print(classification_report(y_test, predictions, target_names=valid_labels))
    cm = confusion_matrix(y_test, predictions, labels=rbf.classes_)
    plt.figure(figsize=(10, 10))
    ax = sn.heatmap(cm, vmax=10000, annot=True, xticklabels=valid_labels, yticklabels=valid_labels)
    plt.savefig('cm_snRNA_random.png')


if __name__ == '__main__':
    seq_data = RNADataset.from_file('/junde/snRNA.fasta')       
    labels_file = pd.read_csv('snRNA_taxid_phylum.csv', index_col=0)
    pprint(labels_file.phylum_name.value_counts())
    label_count = labels_file.phylum_name.value_counts()
    valid_labels = []
    for name, count in zip(label_count.index, label_count.values):
        if count > 1000:
            valid_labels.append(name)
    label_to_idx = {label: i for i, label in enumerate(valid_labels)}
    
    # load training result
    exp_info = torch.load('snRNA_150M_100epoch/checkpoint-99.pth')
    args = exp_info['args']
    alphabet = Alphabet_RNA.RNA(coden_size=args.coden_size)
    model = RESM(alphabet, num_layers=args.num_layers, embed_dim=args.embed_dim, attention_heads=20).cuda()
    model.load_state_dict(exp_info['model'])
    
    # construct data loader
    dist_misc.init_distributed_mode(args)
    num_tasks = dist_misc.get_world_size()
    global_rank = dist_misc.get_rank()
    batch_index = seq_data.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    sampler_train = DistributedBatchSampler(
        seq_data, batch_index, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    data_loader = torch.utils.data.DataLoader(
        seq_data, collate_fn=MaskedBatchConverter(alphabet=alphabet, truncation_seq_length=99999), num_workers=64)

    # extract embeddings
    extract_embedding(data_loader, model, repr_layers=[30], save_name='150M_snRNA_feature_cache.pd.parquet')

    dataset = pd.read_parquet('150M_snRNA_feature_cache.pd.parquet')
    trainset = dataset.groupby('phylum').sample(500, random_state=0)
    testset = dataset.drop(trainset.index)

    # SVM
    svm(trainset, testset)
    pass
    # linear prob

    # finetune

        


