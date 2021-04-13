#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     AUM
# @Filename:    Dataset.py
# @Author:      mdagar
# @Time:        4/12/21 4:15 PM

import torch
class SST(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Add fake data and index
# random_index - points to be marked fake
class SST_AUM(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, random_index,fake_label):
        self.random_index = random_index
        self.encodings = encodings
        self.labels = labels
        self.fake_label = fake_label

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if idx in self.random_index:
            item['labels'] = torch.tensor(self.fake_label)
        else:
            item['labels'] = torch.tensor(self.labels[idx])
        item['index'] = idx
        return item

    def __len__(self):
        return len(self.labels)