#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp.my_yolox_base import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        self.data_dir = "../datasets/Duck-Farm"
        self.train_ann = "labels_train.json"
        self.val_ann = "labels_val.json"
        self.name = "duck-farm"
        
        self.num_classes = 3
        
        self.max_epoch = 50
        self.data_num_workers = 4
        self.eval_interval = 1
        
