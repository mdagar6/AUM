#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     AUM
# @Filename:    Pipeline.py
# @Author:      mdagar
# @Time:        4/15/21 12:47 AM

import configparser
from Data import Data
from Execution import Execution
import torch
import os
from Model import Model
import random
from Util import Util
from aum import AUMCalculator
from transformers import DistilBertTokenizerFast


#temp for flipping
"""
class SST2_flip(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, update_label_index):
        self.encodings = encodings
        self.labels = labels
        self.update_label_index = update_label_index

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if idx in self.update_label_index:
            item['labels'] = torch.tensor((self.labels[idx])^1)
        else:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
"""

class pipeline:

    @classmethod
    def aum_run(cls, config, seed, fakedata_indices, train, threshold_set_no, noise, dev_dataset):
        bsize = int(config["batch_size"])
        lrate = float(config["learning_rate"])
        optimizer = eval(config["optimizer"])
        fake_label = int(config["fake_label"])
        threshold_percentile = float(config["threshold_percentile"])

        # get Dataset + corresponding preprocessing
        train_dataset = Data.get_dataset(train, config["model"], config["dataset"], True, fakedata_indices,
                                         fake_label)

        # model
        save_dir = config["aum_dir"]
        os.makedirs(save_dir, exist_ok=True)
        
        aum_calculator = AUMCalculator(save_dir, compressed=False)
        n_epochs = int(config["aum_training_epoch"])
        model = Model.get_model(config["model"],int(config["num_classes"])+1)
        model = Execution.training(model, bsize, n_epochs, train_dataset, lrate, optimizer, True, save_dir,
                                   aum_calculator, evaluate = True,test_data = dev_dataset)

        torch.save(train_dataset, os.path.join(save_dir, str(seed)+"_"+str(noise)+"_"+threshold_set_no+"_"+str(threshold_percentile)+"_"+"dataset.pth"))

        # Calculate Threshold 1
        aum_data = Util.csv_to_dataframe(save_dir + "aum_values.csv", fakedata_indices)
        Util.plot_aum(aum_data)
        threshold = Util.calculate_threshold(aum_data, threshold_percentile)
        #temp for flipping
        #threshold = aum_data.aum.quantile(threshold_percentile)


        print("Threshold AUM - "+threshold_set_no, threshold)
        #tmp for flipping
        #bad_data1 = list(aum_data[(aum_data['aum'] < threshold)]["sample_id"])
        
        
        bad_data1 = list(aum_data[(aum_data["fake_data_flag"] == 0) & (aum_data['aum'] < threshold)]["sample_id"])
        print("bad data-1 length:", len(bad_data1))

        os.rename(save_dir + "aum_values.csv", save_dir + str(seed) +"_"+ str(noise)+"_"+str(threshold_percentile)+"_"+"aum_values_"+threshold_set_no+".csv")
        os.rename(save_dir + "full_aum_records.csv", save_dir + str(seed) +"_"+ str(noise)+"_"+str(threshold_percentile)+"_"+"full_aum_records_"+threshold_set_no+".csv")
        return bad_data1

    @classmethod
    def run(cls, config, seed, noise):
        # get data
        train, dev = Data.get_data(config["dataset"], config["dataset_dir"], config["data_file_type"],
                                   config["data_file_sep"], noise)

        # get Dataset + corresponding preprocessing
        train_dataset = Data.get_dataset(train, config["model"], config["dataset"])
        dev_dataset = Data.get_dataset(dev, config["model"], config["dataset"])

        # model
        bsize = int(config["batch_size"])
        n_epochs = int(config["training_epoch"])
        lrate = float(config["learning_rate"])
        optimizer = eval(config["optimizer"])
        model = Model.get_model(config["model"], int(config["num_classes"]))
        model = Execution.training(model, bsize, n_epochs, train_dataset, lrate, optimizer, False, evaluate = True,test_data = dev_dataset)

        # evaluation
        Execution.evaluation(model, dev_dataset, bsize)

        # AUM Calculation - Threshold 1
        num_classes = int(config["num_classes"])
        N = len(train)
        fakedata_count = int(N / (num_classes + 1))

        fakedata_indices_1 = set(random.sample(range(N), fakedata_count))
        temp = set(range(N)) - fakedata_indices_1
        fakedata_indices_2 = set(random.sample(temp, fakedata_count))

        bad_data1 = cls.aum_run(config, seed, fakedata_indices_1, train, "1", noise, dev_dataset)
        bad_data2 = cls.aum_run(config, seed, fakedata_indices_2, train, "2", noise, dev_dataset)
        
        mislabelled_data = set(bad_data1).union(set(bad_data2))
        print("Length of Mislabelled_data from thershold set 1: ", len(bad_data1))
        print("Length of Mislabelled_data from thershold set 2: ", len(bad_data2))
        print("Length of Mislabelled_data: ", len(mislabelled_data))

        #Training on filter data
        clean_index = list(set(range(N)) - set(mislabelled_data))
        train_dataset_clean = torch.utils.data.Subset(train_dataset, clean_index)
        print("length of clean dataset: ", len(train_dataset_clean))

        model = Model.get_model(config["model"],num_classes)
        model = Execution.training(model, bsize, n_epochs, train_dataset_clean, lrate, optimizer, False, evaluate = True,test_data = dev_dataset)

        # evaluation
        Execution.evaluation(model, dev_dataset, bsize)
        
        #Flip labels - Not generic
        """
        print("After flip")
        dbtokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        enc = dbtokenizer(train['sentence'], truncation=True, padding=True, max_length=128)
        flip_dataset = SST2_flip(enc, (torch.Tensor(train['label'])).long(), mislabelled_data)
        model = Model.get_model(config["model"],num_classes)
        model = Execution.training(model, bsize, n_epochs, flip_dataset, lrate, optimizer, False, evaluate = True,test_data = dev_dataset)

        # evaluation
        Execution.evaluation(model, dev_dataset, bsize)
        """







