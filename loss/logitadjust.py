import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()#每个类的样本量频率
        m_list = tau * torch.log(cls_p_list)#都是负的log 频率的值
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)
class LogitAdjust_sam(nn.Module):
    """
    修改后的 LogitAdjust，支持 reduction 参数并修正了符号。
    """
    def __init__(self, cls_num_list, tau=1.0, reduction='mean', weight=None):
        super(LogitAdjust_sam, self).__init__()
        # 确保 cls_num_list 是 FloatTensor
        cls_num_list = torch.FloatTensor(cls_num_list)
        
        # 计算 log 先验概率
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        
        # 将 m_list 注册为 buffer，这样它会自动移动到 .cuda() 或 .cpu()
        self.register_buffer('m_list', m_list.view(1, -1))
        
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        # x 是模型的原始输出 logits
        # target 是真实标签
        
        # 修正符号：减去 m_list (一个负向量)，等同于加上一个正的调整项
        # 这会提升少数类的 logits
        x_m = x - self.m_list
        
        # 将 self.reduction 参数传递给 F.cross_entropy
        return F.cross_entropy(x_m, target, weight=self.weight, reduction=self.reduction)
class GroupLogitAdjust(nn.Module):

    def __init__(self, cls_num_list, groups, tau=1, weight=None, others_flag = True):
        super(GroupLogitAdjust, self).__init__()
        groups_m_list = self.get_group_m_list(groups, cls_num_list, tau,others_flag)
        self.groups_m_list = groups_m_list
        self.weight = weight
        self.others_flag = others_flag

    def forward(self, x, target, group_index):
        m_list_g = self.groups_m_list[group_index]
        x_m = x + m_list_g
        return F.cross_entropy(x_m, target, weight=self.weight)

    def get_group_m_list(self, groups,cls_num_list,tau,others_flag):
        groups_m_list = []
        total_num = sum(cls_num_list)
        for g,group_g in enumerate(groups):
            cls_num_list_g = []
            for label_i in group_g:
                cls_num_label_i = cls_num_list[label_i]
                cls_num_list_g.append(cls_num_label_i)
            if others_flag:
                others_num = total_num - sum(cls_num_list_g)
                cls_num_list_g.append(others_num)
                if sum(cls_num_list_g) != total_num:
                    print('error num list!!')
            cls_num_list_g = torch.cuda.FloatTensor(cls_num_list_g)
            cls_p_list_g = cls_num_list_g / total_num
            m_list_g = tau * torch.log(cls_p_list_g)
            groups_m_list.append(m_list_g.view(1,-1))

        return groups_m_list


