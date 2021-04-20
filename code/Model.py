#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     AUM
# @Filename:    model.py
# @Author:      mdagar
# @Time:        4/14/21 5:37 AM

from transformers import DistilBertForSequenceClassification

class Model:
    @classmethod
    def get_model(cls, model_name, no_classes):
        return eval("cls."+model_name+"(no_classes)")
    
    @classmethod
    def distilbert(cls, no_classes):
        return DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=no_classes)
