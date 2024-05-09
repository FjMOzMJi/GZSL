import json
import math
import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def read_decomposed_intents(info_path):
    with open(info_path / 'concepts.json') as f:
        concepts = json.load(f)
    with open(info_path / 'actions.json') as f:
        actions = json.load(f)
    return concepts, actions


def read_zeroshot_intents(data_path):
    zeroshot_intents = None
    zeroshot_path = data_path / 'zeroshot_intents.json'
    if zeroshot_path.exists():
        with open(zeroshot_path) as f:
            zeroshot_intents = json.load(f, object_pairs_hook=OrderedDict)
    return zeroshot_intents


def read_split_data_step1(data_path, intents, seed):
    zeroshot_intents = read_zeroshot_intents(data_path)
    if zeroshot_intents is None:
        random.shuffle(intents)
        unseen_len = math.ceil(len(intents) * 0.25)
        zeroshot_intents = intents[-unseen_len:]
        all_df = pd.read_csv(data_path / 'all.csv')
        seen_df = all_df[~all_df.intents.isin(zeroshot_intents)]
        unseen_df = all_df[all_df.intents.isin(zeroshot_intents)]
        train_df, test_df = train_test_split(
            seen_df,
            test_size=0.3,
            random_state=seed,
            stratify=seen_df.intents
        )
        train_df, dev_df = train_test_split(
            train_df,
            test_size=0.1,
            random_state=seed,
            stratify=train_df.intents
        )
        test_df = pd.concat([test_df, unseen_df])
    else:
        train_df = pd.read_csv(data_path / 'train.csv')
        dev_df = pd.read_csv(data_path / 'dev.csv')
        test_df = pd.read_csv(data_path / 'test.csv')
    return zeroshot_intents, train_df, dev_df, test_df


def read_split_data_step2(data_path, intents, seed):
    zeroshot_intents = read_zeroshot_intents(data_path)
    random.shuffle(intents)
    unseen_len = math.ceil(len(intents) * 0.25)
    zeroshot_intents = intents[-unseen_len:]
    all_df = pd.read_csv(data_path / 'all.csv')
    seen_df = all_df[~all_df.intents.isin(zeroshot_intents)]
    unseen_df = all_df[all_df.intents.isin(zeroshot_intents)]
    original_train_df, test_df = train_test_split(
        seen_df,
        test_size=0.3,
        random_state=seed,
        stratify=seen_df.intents
    )

    test_df = pd.concat([test_df, unseen_df])
    train_df = pd.read_csv(data_path / 'gen_data_flitted.csv')

    sampled_unseen_df = train_df[train_df.intents.isin(zeroshot_intents)]
    
    sampled_seen_df = pd.DataFrame()
    for intent in  intents[:-unseen_len]:
        intent_records = original_train_df[original_train_df['intents'] == intent]
        if len(intent_records) > 50:
            sampled_records = intent_records.sample(n=50, random_state=seed)
        else:
            sampled_records = intent_records 
        sampled_seen_df = pd.concat([sampled_seen_df, sampled_records])

    train_df = pd.concat([sampled_unseen_df, sampled_seen_df])
    try:
        train_df, dev_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=seed,
            stratify=train_df.intents
        )
    except:
        train_df, dev_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=seed,
        )

    return zeroshot_intents, train_df, dev_df, test_df


def read_intent_info(info_path, description_type,dataset_name):
    with open(info_path / f'descriptions/{description_type}.json') as f:
        intent_with_desc = json.load(f, object_pairs_hook=OrderedDict)
        intents = list(intent_with_desc.keys())
        descriptions = {i: desc for i, desc in intent_with_desc.items()}
    return intents, descriptions


def read_intent_similarity_matrix(info_path, matrix_name, intents):
    with open(info_path / matrix_name / 'raws.json') as f:
        raw_intents = json.load(f)
    intent_idxs = [raw_intents.index(intent) for intent in intents]
    matrix = np.loadtxt(info_path / matrix_name / 'similarity.txt', delimiter=',')
    matrix = matrix\
        .take(intent_idxs, axis=0)\
        .take(intent_idxs, axis=1)
    return matrix


def read_uttr_similarity_matrix(data_path, matrix_name, train_indexes):
    matrix = np.loadtxt(data_path / 'uttr_similarity' / matrix_name, delimiter=',', dtype=int)
    train_indexes_inversed = {source_idx: tr_idx for tr_idx, source_idx in enumerate(train_indexes)}
    tr_indexes_set = set(train_indexes)
    similar_utterances = [
        [train_indexes_inversed[j] for j in matrix[source_idx] if j in tr_indexes_set]
        for source_idx in train_indexes
    ]
    return similar_utterances


def draw_matrix(matrix, labels):
    fig, ax = plt.subplots()
    _ = ax.imshow(matrix, cmap='hot', interpolation='nearest')
    ax.tick_params(axis='x', which='major', labelsize=5)
    ax.tick_params(axis='y', which='major', labelsize=5)
    ax.set_xticks(np.arange(0, len(labels), 1))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(0, len(labels), 1))
    ax.set_yticklabels(labels)
    fig.tight_layout()
    fig.set_size_inches((15, 15))
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%H_%M_%S")
    plt.savefig(os.path.join('plots', f'sgd_{dt_string}.png'))
