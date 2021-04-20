#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     AUM
# @Filename:    Data.py
# @Author:      mdagar
# @Time:        4/14/21 5:38 AM

import pandas as pd
from datasets import Dataset
from Custom_Dataset import SST2, SST2_AUM
import torch
from transformers import DistilBertTokenizerFast
import random


class Data:
    dbtokenizer = None

    @classmethod
    def get_data(cls, dataset_name, dataset_dir, data_file_type, data_file_sep, noise = None):
        loc = dataset_dir + dataset_name + "/"
        train = pd.read_csv(loc + "train" + data_file_type, sep=data_file_sep, header='infer', index_col=0)
        # add noise
        if noise:
            labels = set(train['label'])
            N = len(train)
            noise = int(noise * N)
            noise_index = set(random.sample(range(N), noise))
            train['label'] = [random.sample(labels - set([i]), 1)[0] if idx in noise_index else i for idx, i in
                              enumerate(train["label"])]
        train = Dataset.from_pandas(train)

        dev = Dataset.from_pandas(
            pd.read_csv(loc + "dev" + data_file_type, sep=data_file_sep, header='infer', index_col=0))
        return train, dev

    @classmethod
    # dataset format required by model
    def get_dataset(cls, data, model, dataset, aum=False, fake_index=None, fake_label=None):
        if aum:
            return eval("cls." + model + "_" + dataset + "_AUM(data, fake_index, fake_label)")
        else:
            return eval("cls." + model + "_" + dataset + "(data)")

    @classmethod
    def distilbert_SST2(cls, data):
        cls.dbtokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        enc = cls.dbtokenizer(data['sentence'], truncation=True, padding=True, max_length=128)
        dataset = SST2(enc, (torch.Tensor(data['label'])).long())
        return dataset

    @classmethod
    def distilbert_SST2_AUM(cls, data, fake_index, fake_label):
        if not cls.dbtokenizer:
            cls.dbtokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        enc = cls.dbtokenizer(data['sentence'], truncation=True, padding=True, max_length=128)
        dataset = SST2_AUM(enc, (torch.Tensor(data['label'])).long(), fake_index, fake_label)
        return dataset

