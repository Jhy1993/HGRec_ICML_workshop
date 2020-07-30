#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
"""

"""

# adj_mat: UU.mat, UIU.mat, IUI.mat, IBI.mat
# fea_mat: U_fea.mat, I_fea.mat
# train/test: train.rating, test.rating, test.negative


import random
from collections import defaultdict

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


class Data_loader():
    def __init__(self,
                 cfg
                 # path='./Data/ml-100k/',
                 # U_MP=['UU', 'UMU'],
                 # U_Fea='UMG',
                 # I_MP=['MUM', 'MGM'],
                 # I_Fea='MG'
                 ):
        self.path = cfg.path

        train_file = self.path + 'ml-100k.train.rating'  # os.path.join(path, 'train.rating')
        test_file = self.path + 'ml-100k.test.rating'  # os.path.join(path, 'test.rating')
        # test_neg_file = os.path.join(path, 'test.negative')
        # load data

        self.train_data, self.train_user_dict = self.load_rating(train_file)
        self.test_data, self.test_user_dict = self.load_rating(test_file)

        self.test_set = self.test_user_dict
        self.train_items = self.train_user_dict

        self.exist_users = self.train_user_dict.keys()

        self.get_info()  #
        self.U_mp_list = self.load_MP(cfg.U_MP[1:])
        self.I_mp_list = self.load_MP(cfg.I_MP[1:])
        self.U_fea = self.load_fea(cfg.U_Fea)
        self.I_fea = self.load_fea(cfg.I_Fea)

   

        self.UIUI = self.load_UI()

    def load_UI(self):
        fp = self.path + 'UM.mat'
        mat = sio.loadmat(fp)['UM']
        UI_adj_mat = np.zeros([self.n_users + self.n_items, self.n_users + self.n_items])
        mat = mat.todense()
        UI_adj_mat[0:self.n_users, self.n_users:] = mat
        UI_adj_mat[self.n_users:, :self.n_users] = mat.T
        print(f"UIUI_adj_mat: {UI_adj_mat.shape}")

        res = UI_adj_mat + np.diag(np.ones(UI_adj_mat.shape[0]))

        res_ = np.dot(np.diag(1 / np.sum(res, axis=1)),
                      res)
        return res

    def load_fea(self, fea):
        fp = self.path + fea + '.mat'
        mat = sio.loadmat(fp)[fea]
        print(f"Load Fea for {fea[0]}, shape: {mat.shape}")
        return mat

    def load_MP(self, mp_list):
        mat_list = []
        fp_list = [self.path + i + '.mat' for i in mp_list]
        for mp, mp_path in zip(mp_list, fp_list):
            if mp == 'no':
                return None
            else:
                print(mp, mp_path)
                tmp = sio.loadmat(mp_path)[mp]
                print(
                    f"Load {mp[0]} MP mat: {mp}, shape: {tmp.shape}, sparse: {np.sum(tmp) / (tmp.shape[0] * tmp.shape[1])}")
                if sp.issparse(tmp):
                    print(f"{mp} is sp and to dense")
                    tmp = tmp.todense()
            mat_list.append(tmp)
        return mat_list

    def get_info(self):
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)
        self.n_users = max(
            max(self.train_data[:, 0]), max(self.test_data[:, 0])
            
        ) + 1   
        self.n_items = max(
            max(self.train_data[:, 1]), max(self.test_data[:, 1])
        ) + 1
        print(
            f"data: {self.path}, n_users: {self.n_users}, n_items: {self.n_items}, n_train: {self.n_train}, n_test: {self.n_test}")

    def load_rating(self, path):
        k = 0
        # user_dict = dict()
        user_dict = defaultdict(list)
        inter_list = list()
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split('\t')
                inters = [int(i) for i in line]
                user = inters[0]
                item_list = list(set(inters[1:]))
                for item in item_list:
                    inter_list.append([user, item])
                    user_dict[user].append(item)
        return np.array(inter_list), user_dict

    def generate_train_batch(self, batch_size):
        users, pos_items, neg_items = self._generate_train_cf_batch(batch_size)
       
        batch_data = {}
        batch_data['users'] = users
        batch_data['pos_items'] = pos_items
        batch_data['neg_items'] = neg_items
        return batch_data

    def generate_test_feed_dict(self, model, user_batch, item_batch, drop_flag=False):
        
        feed_dict = {
            model.users: user_batch,
            model.pos_items: item_batch
        }
        return feed_dict

    def generate_train_feed_dict(self, model, batch_data):
        feed_dict = {
            model.users: batch_data['users'],
            model.pos_items: batch_data['pos_items'],
            model.neg_items: batch_data['neg_items']
        }

        return feed_dict

    def _generate_train_cf_batch(self, batch_size):
        if batch_size <= self.n_users:
            users = random.sample(self.exist_users, batch_size)
        else:
            
            users = [random.choice(self.exist_users) for _ in range(batch_size)]

        def sample_pos_for_u(u, num):
            pos_items = self.train_user_dict[u]
            sample_pos_items = []
            while True:
                if len(sample_pos_items) == num:
                    break
                p_item_id = np.random.randint(0, len(pos_items), 1)[0]
                p_item = pos_items[p_item_id]
                if p_item not in sample_pos_items:
                    sample_pos_items.append(p_item)
            return sample_pos_items

        def sample_neg_for_u(u, num):
            sample_neg_items = []
            while True:
                if len(sample_neg_items) == num:
                    break
                n_item_id = np.random.randint(0, self.n_items, 1)[0]
                if n_item_id not in self.train_user_dict[u] and n_item_id not in sample_neg_items:
                    sample_neg_items.append(n_item_id)
            return sample_neg_items

        pos_items = []
        neg_items = []
        for u in users:
            pos_items += sample_pos_for_u(u, 1)
            neg_items += sample_neg_for_u(u, 1)
        return users, pos_items, neg_items


if __name__ == "__main__":
    from config import config_ml_100k as cfg

    data = Data_loader(cfg)
    UMU = data.U_mp_list[0]
    MUM = data.I_mp_list[0]
    print(np.sum(UMU))
