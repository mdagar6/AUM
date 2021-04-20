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


class pipeline:

    @classmethod
    def aum_run(cls, config, seed, fakedata_indices, train, threshold_set_no, noise):
        checkpoint_dir = config["checkpoint_dir"]
        bsize = int(config["batch_size"])
        lrate = float(config["learning_rate"])
        optimizer = eval(config["optimizer"])
        fake_label = int(config["fake_label"])

        # get Dataset + corresponding preprocessing
        train_dataset = Data.get_dataset(train, config["model"], config["dataset"], True, fakedata_indices,
                                         fake_label)

        # model
        save_dir = config["aum_dir"]
        aum_calculator = AUMCalculator(save_dir, compressed=True)
        n_epochs = int(config["aum_training_epoch"])
        model = Model.get_model(config["model"],int(config["num_classes"])+1)
        model = Execution.training(model, bsize, n_epochs, train_dataset, lrate, optimizer, True, save_dir,
                                   aum_calculator)

        # save model
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, str(seed) +"_"+ str(noise)+   "model_with_AUM_1.pth"))
        torch.save(train_dataset, os.path.join(checkpoint_dir, str(seed) +"_"+ str(noise)+   "dataset_AUM_1.pth"))

        # Calculate Threshold 1
        aum_data = Util.csv_to_dataframe(save_dir + "aum_values.csv", fakedata_indices)
        Util.plot_aum(aum_data)
        threshold = Util.calculate_threshold(aum_data, float(config["threshold_percentile"]))
        print("Threshold AUM - "+threshold_set_no, threshold)
        bad_data1 = list(aum_data[(aum_data["fake_data_flag"] == 0) & (aum_data['aum'] < threshold)]["sample_id"])
        print("bad data-1 length:", len(bad_data1))

        os.rename(save_dir + "aum_values.csv", save_dir + str(seed) +"_"+ str(noise)+   "aum_values_"+threshold_set_no+".csv")
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
        model = Execution.training(model, bsize, n_epochs, train_dataset, lrate, optimizer, False)

        # save the model
        checkpoint_dir = config["checkpoint_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, str(seed) +"_"+ str(noise)+   "model_without_AUM.pth"))

        # evaluation
        Execution.evaluation(model, dev_dataset, bsize)

        # AUM Calculation - Threshold 1
        num_classes = int(config["num_classes"])
        N = len(train)
        fakedata_count = int(N / (num_classes + 1))
        fakedata_per_class = fakedata_count // num_classes

        # datapoint indices from different classes
        classes = [set() for i in range(num_classes)]
        for idx, i in enumerate(train['label']):
            classes[i].add(idx)
        # for two threshold sets
        fakedata_indices_1 = set()
        fakedata_indices_2 = set()

        for i in classes:
            fakedata_indices_1=fakedata_indices_1.union(set(random.sample(i, fakedata_per_class)))
            temp = i - fakedata_indices_1
            fakedata_indices_2=fakedata_indices_2.union(set(random.sample(temp, fakedata_per_class)))

        bad_data1 = cls.aum_run(config, seed, fakedata_indices_1, train, "1", noise)
        bad_data2 = cls.aum_run(config, seed, fakedata_indices_2, train, "2", noise)

        mislabelled_data = set(bad_data1).union(set(bad_data2))
        print("Length of Mislabelled_data from thershold set 1: ", len(bad_data1))
        print("Length of Mislabelled_data from thershold set 2: ", len(bad_data2))
        print("Length of Mislabelled_data: ", len(mislabelled_data))

        #Training on filter data
        clean_index = list(set(range(N)) - set(mislabelled_data))
        train_dataset_clean = torch.utils.data.Subset(train_dataset, clean_index)
        print("length of clean dataset: ", len(train_dataset_clean))

        model = Model.get_model(config["model"],num_classes)
        model = Execution.training(model, bsize, n_epochs, train_dataset_clean, lrate, optimizer, False)

        # save the model
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, str(seed) +"_"+ str(noise)+   "model_on_clean_data.pth"))

        # evaluation
        Execution.evaluation(model, dev_dataset, bsize)

        """
        # get Dataset + corresponding preprocessing
        train_dataset = Data.get_dataset(train, config["model"], config["dataset"], True, fakedata_indices_1, fake_label)

        # model
        save_dir =  config["aum_dir"]
        aum_calculator = AUMCalculator(save_dir, compressed=True)
        n_epochs = int(config["aum_training_epoch"])
        model = Model.get_model(config["model"])
        model = Execution.training(model, bsize, n_epochs, train_dataset, lrate, optimizer, True, save_dir, aum_calculator)

        # save model
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, str(seed) + "model_with_AUM_1.pth"))
        torch.save(train_dataset, os.path.join(checkpoint_dir, str(seed) + "dataset_AUM_1.pth"))

        #Calculate Threshold 1
        aum_data = Util.csv_to_dataframe(save_dir+"aum_values.csv", fakedata_indices_1)
        Util.plot_aum(aum_data)
        threshold = Util.calculate_threshold(aum_data, config["threshold_percentile"])
        print("Threshold AUM - 1: ", threshold)
        bad_data1 = list(aum_data[(aum_data["fake_data_flag"] == 0) & (aum_data['aum'] < threshold)]["sample_id"])
        print("bad data-1 length:", len(bad_data1))

        os.rename(save_dir+"aum_values.csv", save_dir + str(seed) +"aum_values_1.csv")
        """

