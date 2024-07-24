"""
Implementation for the Memory Bank for pixel-level feature vectors
"""

import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from collections import deque
import math


def funcget_group_centroid(logits, labels, centroid):
    # data_t_unl = next(data_iter_t_unl)
    groups = {}
    for x, y in zip(labels, logits):
        group = groups.get(x.item(), [])
        group.append(y)
        groups[x.item()] = group
    for key in groups.keys():
        groups[key] = torch.stack(groups[key]).mean(dim=0)
    if centroid != None:
        for k, v in centroid.items():
            if groups != None and k in groups:
                centroid[k] = 0.99 * v + 0.01 * groups[k]
            else:
                centroid[k] = v
    if groups != None:
        for k, v in groups.items():
            if k not in centroid:
                centroid[k] = v
    return centroid



class FeatureLabelMemoryBank:
    def __init__(self, dataset, num_classes=65, max_features_per_class=512, feature_dim=512):
        self.num_classes = num_classes
        self.max_features_per_class = max_features_per_class
        self.feature_dim = feature_dim
        self.proto_s = Prototype(dataset=dataset['name'], C=num_classes, dim=512)
        # 为每个类别创建一个队列
        self.feature_queues = {label: deque(maxlen=max_features_per_class) for label in range(num_classes)}

    def insert_feature_label(self, feature, label):
        # 插入新的特征和标签到对应类别的队列
        feature = F.normalize(feature, dim=0)
        self.feature_queues[label.item()].append(feature)


    def get_all_features_labels(self):
        all_features, all_labels = [], []
        for label, queue in self.feature_queues.items():
            all_features.extend(queue)
            all_labels.extend([label] * len(queue))
        return torch.stack(all_features).cuda(), torch.tensor(all_labels, dtype=torch.int64).cuda()

    def get_queue_lengths(self):
        for label, queue in self.feature_queues.items():
            print(label, len(queue))

    def compute_class_means_matrix(self):
        class_means_matrix = []

        for label, queue in self.feature_queues.items():
            if len(queue) > 0:
                # 计算每个类别的特征平均值
                class_features = torch.stack(list(queue))
                class_mean = torch.mean(class_features, dim=0)
                class_means_matrix.append(class_mean)
        return torch.stack(class_means_matrix)


class Prototype:
    def __init__(self, dataset, C=65, dim=512, m=0.99):
        self.mo_pro = torch.zeros(C, dim).cuda()
        self.batch_pro = torch.zeros(C, dim).cuda()
        self.m = m
    @torch.no_grad()
    def update(self, feats, lbls, i_iter, norm=False):
        if i_iter < 20:
            momentum = 0
        else:
            momentum = self.m
        feats = F.normalize(feats)
        for i_cls in torch.unique(lbls):
            feats_i = feats[lbls == i_cls, :]
            feats_i_center = feats_i.mean(dim=0, keepdim=True)
            self.mo_pro[i_cls, :] = self.mo_pro[i_cls, :] * \
                momentum + feats_i_center * (1 - momentum)
            self.batch_pro[i_cls, :] = feats_i_center
        if norm:
            self.mo_pro = F.normalize(self.mo_pro)
            self.batch_pro = F.normalize(self.batch_pro)