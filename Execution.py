#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     AUM
# @Filename:    Execution.py
# @Author:      mdagar
# @Time:        4/12/21 4:14 PM

import torch
from aum import AUMCalculator
from tqdm.notebook import tqdm
import os

class Execution:
    @staticmethod
    def evaluation(model, test_data, bsize):
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=bsize, shuffle=True)
        model.eval()
        acc = 0.0
        for batch in test_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            outputs = model(input_ids, attention_mask, labels=labels)
            acc += torch.sum(outputs.logits.argmax(dim=-1) == labels)
        print("Test Acc: ", acc / len(test_data))
        
    @staticmethod
    def training(model, bsize, n_epochs, train_data, test_data, lrate, optimizer, with_aum, aum_path, aum_calculator, evaluate = False):
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=bsize, shuffle=True)
        model.cuda()
        opt = optimizer(params=model.parameters(), lr=lrate)
        model.train()
        for i in tqdm(range(n_epochs)):
            total_train_loss = 0.0
            acc = 0.0
            for batch in tqdm(dataloader, leave=False):
                label = batch["labels"].cuda()
                data = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()

                outputs = model(data, attention_mask=attention_mask, labels=label)
                acc += torch.sum(outputs.logits.argmax(dim=1) == label)
                loss = outputs.loss

                if with_aum:
                    index = batch["index"].cuda()
                    records = aum_calculator.update(outputs.logits, label, index)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_train_loss += loss

            print("Train_Loss:", total_train_loss / len(dataloader), "Acc: ", acc / len(train_data))
            if(evaluate):
                evaluation(model, dev_data, bsize)
        if with_aum:
            if os.path.exists(aum_path):
                os.remove(aum_path)
            aum_calculator.finalize()

        return model

    
