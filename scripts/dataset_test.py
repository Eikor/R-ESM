import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import sys
import esm
import torch
torch.set_printoptions(profile="full")
from data import MaskedBatchConverter, DistributedBatchSampler, Alphabet_RNA, RNADataset

from multiprocessing import Pool
import threading
from tqdm import tqdm
import os


def check_ddict(data_dict):

    for k,v in data_dict.items():
        assert not torch.isnan(v).any(), 'nan in {}'.format(k)
        assert not torch.isinf(v).any(), 'inf in {}'.format(k)
    non_padding_length = torch.sum(data_dict['padding_mask'])
    loss_length = torch.sum(data_dict['loss_mask'])
    if int(non_padding_length) == 0:
        raise AssertionError('data length should not be zero!')
    if int(loss_length) == 0:
        raise AssertionError('loss lenghth should not be zero!')


def test(indices, fail_path='check0129.log',assertion_path='assertion_error.log'):
    fails_writer = open(fail_path,'a')
    assertion_writer = open(assertion_path,'a')
    try:
        a,b,data = batch_converter([dataset[i] for i in indices])
        check_ddict(data)
    except AssertionError:
        print('assertion error happens')
        assertion_writer.write([dataset[i] for i in indices][0][0]+'\n')
    except Exception as ex:
        fails_writer.write([dataset[i] for i in indices][0][0]+'\n')

def test_data(indices):
    try:
        batch_converter([dataset[i] for i in indices])
        return None
    except Exception as ex:
        return [dataset[i] for i in indices][0][0]

def setcallback(x):
    if x is None:
        return
    else:
        print('find exception writing')
        with open('new_fail_labels.txt', 'a+') as ff:
            ff.write(x+'\n')


if __name__ == "__main__":
    dataset = RNADataset.from_file(sys.argv[1])

    batch_index = dataset.get_batch_indices(1024, extra_toks_per_seq=1)
    batch_sampler = DistributedBatchSampler(
            dataset,
            batch_index,
            num_replicas=1,
            rank=0,
            shuffle=True
    )

    alphabet = Alphabet_RNA.RNA(coden_size=3)
    batch_converter = MaskedBatchConverter(alphabet, 1022)
    data_loader = torch.utils.data.DataLoader(dataset,
                                        batch_sampler=batch_sampler,
                                        num_workers=0,
                                        pin_memory=True)
    data_loader_iter = iter(data_loader)

    print(len(dataset))
    print('finish initialization>\n start testing')

    fail_log_path = sys.argv[2]
    assertion_log_path = sys.argv[3]

    print(len(batch_sampler))
    if os.path.exists(fail_log_path):
        os.remove(fail_log_path)
    if os.path.exists(assertion_log_path):
        os.remove(assertion_log_path)


    pool = Pool()

    pbar = tqdm(total=len(batch_sampler))
    update = lambda *args: pbar.update()
    for indices in batch_sampler:
        pool.apply_async(func=test, args=(indices,fail_log_path,assertion_log_path),callback=update)
    pool.close()
    pool.join()

