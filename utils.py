data_dir = 'CelebA/Imgs'
identity_file = 'CelebA/Anno/identity_CelebA.txt'

import os
from glob import glob
import matplotlib.pyplot as plt
from collections import Counter
import random

def remove_format(file_name):
    f = os.path.basename(file_name)
    return f.split('.')[0]

def count_indentity(identity_file):

    identity_counter = Counter()
    identity_dict = dict()

    with open(identity_file, 'r') as f:
        for i, line in enumerate(f):
            file, identity = line.split()
            identity = int(identity)
            full_path = os.path.join('CelebA/Img/img_align_celeba', file)
            # identity_dict[remove_format(file)] = identity
            identity_dict[full_path] = identity
            identity_counter[int(identity)] += 1
            
    print(f'{i+1} files')

    return identity_counter, identity_dict

def plot_statistics(identity_counter):

    counts = list()

    for key in identity_counter.keys():
        counts.append(identity_counter[key])
    
    b_max, b_min = max(counts), min(counts)
    
    plt.figure()
    plt.title('Face Indentity Num Distribution')
    plt.hist(counts, bins=b_max - b_min, rwidth=0.8)
    plt.savefig('face_id_hist.jpg')

    print(f'max {b_max} / min {b_min}')

    print(f'{sum(counts)} files')

    print(f'{len(identity_counter.keys())} indentities')

    simple_mean = 1 / len(identity_counter.keys())
    biased_mean_max = max(counts) / sum(counts)
    biased_mean_min = min(counts) / sum(counts)

    print(f'Minimum Random Accuracy {biased_mean_min * 100:02.4f}%')
    print(f'Average Random Accuracy {simple_mean * 100:02.4f}%')
    print(f'Maximum Random Accuracy {biased_mean_max * 100:02.4f}%')

    return

def reduce_counter(identity_counter, min_count):
    c = Counter({specker: count for specker, count \
                 in identity_counter.most_common() if count >= min_count})
    print(f'n >= {min_count} speakers: {len(c)}')
    return c

def reduce_indentity_dict(reduced_identity_counter, identity_dict):
    reduced_dict = dict()

    keys = identity_dict.keys()

    for key in keys:
        if identity_dict[key] in reduced_identity_counter:
            reduced_dict[key] = identity_dict[key]

    return reduced_dict

def get_renumbering_identities_dict(identity_counter):
    old_keys = sorted(identity_counter.keys())
    old2new = {key: i for i, key in enumerate(old_keys)}

    print(f'Original Speaker Num {max(old_keys)} ' + \
          f'=> Reduced Speaker Num {len(old2new.keys())}')

    return old2new

def split_dataset(dataset, ratio=0.02):
    
    random.shuffle(dataset)

    test_length = int(ratio * len(dataset))

    test_dataset = dataset[:test_length]
    train_dataset = dataset[test_length:]

    return train_dataset, test_dataset
