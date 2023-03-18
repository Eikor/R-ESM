# Script for preparing snRNA data
import sys
import os
from tqdm import tqdm


def read_labels(filename):
    fail_labels = []
    with open(filename,'r') as ff:
        for line in ff:
            fail_labels.append(line.strip())
    return fail_labels

if __name__ == '__main__':
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
                
            # if line_idx % 10000 == 0:
            #     print('processed line {}'.format(line_idx))
