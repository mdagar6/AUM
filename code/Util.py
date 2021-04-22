#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     AUM
# @Filename:    Util.py
# @Author:      mdagar
# @Time:        4/14/21 5:37 AM

import seaborn
import re
import pandas as pd
from  matplotlib import pyplot

class Util:

    @classmethod
    # also add flag for fake data
    def csv_to_dataframe(cls, path, fake_data_index):
        aum = pd.read_csv(path, sep=',', header='infer')
        # create flag for fake data
        fake_data_flag = [1 if i in set(fake_data_index) else 0 for i in aum['sample_id']]  # this takes alot of time
        aum["fake_data_flag"] = fake_data_flag
        return aum

    @classmethod
    def plot_aum(cls, aum_data):
        seaborn.set(style='ticks')
        fake = [0, 1]
        fg = seaborn.FacetGrid(data=aum_data, hue='fake_data_flag', hue_order=fake, aspect=1.61)
        fg.map(pyplot.scatter, 'aum', 'sample_id').add_legend()

    @classmethod
    def calculate_threshold(cls, aum_data, percentile):
        return aum_data[aum_data['fake_data_flag'] == 1].aum.quantile(percentile)
