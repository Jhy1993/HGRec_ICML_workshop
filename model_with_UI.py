#!/usr/bin/env python
# -*- coding: utf-8 -*-




import os
import random
from collections import Counter, defaultdict, namedtuple, deque
import numpy as np
import scipy.io as sio
import tensorflow as tf


class HGRec_with_UI:
    """
    0820 version
    """

    def __init__(self, cfg,
                 U_adj_list=None,
                 I_adj_list=None,
                 UIUI=None,
                 have_feature=None):

        # have_features = [U_fea_mat, I_fea_mat]

        self.gnn_type = cfg.gnn_type
        self.depth = cfg.depth  # depth for all aggregator
        self.u_gnn_depth = cfg.u_gnn_depth
        self.i_gnn_depth = cfg.i_gnn_depth
        self.co_attention_transform = cfg.co_attention_transform
        # hyper para
        self.emb_dim = cfg.emb_dim
        self.l2_reg = cfg.l2_reg
        self.lr = cfg.lr
        self.max_degree = cfg.max_degree

        self.U_adj_list = U_adj_list
        self.I_adj_list = I_adj_list
        # if [ui, ui]
        self.if_UIUI = cfg.if_UIUI
        self.UIUI = UIUI
        self.UIUI_depth = cfg.UIUI_depth

        self.have_feature = have_feature
        # multi_embed_1.get_shape().as_list()[1]
        if U_adj_list is not None:
            self.num_mp_1 = len(U_adj_list) + 1 if cfg.if_UIUI else len(U_adj_list)
        else:
            self.num_mp_1 = 1 if cfg.if_UIUI else 0
            # multi_embed_2.get_shape().as_list()[1]
        if I_adj_list is not None:
            self.num_mp_2 = len(I_adj_list) + 1 if cfg.if_UIUI else len(I_adj_list)
        else:
            self.num_mp_2 = 1 if cfg.if_UIUI else 0  # 0 是的时候就没有任意信息...

        if self.have_feature is None:
            self.num_dim_1 = self.emb_dim
            self.num_dim_2 = self.emb_dim
        else:
            self.num_dim_1 = have_feature[0].shape[1]
            self.num_dim_2 = have_feature[1].shape[1]

        self.n_users = 943  # UIUI.shape[0]
        self.n_items = 1682  # UIUI.shape[1]
        print(f"Model: HGRec including UI, (U-mp: {self.num_mp_1}, M-mp: {self.num_mp_2} with co-attention)")
        # place hoders
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        # drop
        # drop some values in embedding, e.g., [0.1] * depth
        self.ffd_drop = tf.placeholder(tf.float32, shape=[None])
        # drop for att coef, actually drop some neighbors when aggregating e.g., [0.1]* depth
        self.coef_drop = tf.placeholder(tf.float32, shape=[None])
        # self.holders = self.build_holders()
        # self.all_weights = self.init_vars()
        self.build_vars()

        self.dual_gnn_update_embed()
        # self.multi_item_embed = tf.expand_dims(self.init_item_embed, axis=1)
        # self.multi_user_embed = tf.expand_dims(self.init_user_embed, axis=1)

        # ========== get multi embed for user/item
        # [bs, num_mp, dim]
        # user_gnn_embedding
        print(
            f"User embed: {self.multi_user_embed.get_shape().as_list()}, "
            f"Item Embed: {self.multi_item_embed.get_shape().as_list()}")
        self.u_g_emb = tf.nn.embedding_lookup(self.multi_user_embed, self.users)
        self.pos_i_g_emb = tf.nn.embedding_lookup(self.multi_item_embed, self.pos_items)
        self.neg_i_g_emb = tf.nn.embedding_lookup(self.multi_item_embed, self.neg_items)
        # co attention
        self.pos_u_att_emb, self.pos_i_att_emb, self.neg_u_att_emb, self.neg_i_att_emb = self.co_attention(
            transform=self.co_attention_transform)
        # bpr_loss
        self.mf_loss, self.emb_loss = self.bpr(self.pos_u_att_emb, self.pos_i_att_emb,
                                               self.neg_u_att_emb, self.neg_i_att_emb)
        self.loss = self.mf_loss + self.l2_reg * self.emb_loss
        self.emb_loss_with_coef = self.l2_reg * self.emb_loss
        # 基于gnn Embedding -> co-attention -> pred
        self.pred_op = self.batch_pred(self.u_g_emb, self.pos_i_g_emb)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        # for NGCF
        # self.batch_ratings = self.pred_op

    def concat(self):
        pass

    def mean(self):
        return tf.reduce_mean(self.u_g_emb, axis=1), tf.reduce_mean(self.pos_i_g_emb, axis=1), tf.reduce_mean(
            self.u_g_emb, axis=1), tf.reduce_mean(self.neg_i_g_emb, axis=1)

    def train(self, sess, feed_dict):
        return sess.run([self.opt, self.loss, self.mf_loss, self.emb_loss_with_coef],
                        feed_dict=feed_dict)

    def bpr(self, pu, pi, nu, ni):
        # todo 这个版本只能一正一负样本， 如果想用neumf的试验结果要改成支持一正多负的
        # pos_user, pos_item, neg_user, neg_item
        pos_pred = tf.reduce_sum(tf.multiply(pu, pi),
                                 axis=1)
        neg_pred = tf.reduce_sum(tf.multiply(nu, ni),
                                 axis=1)

        _mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_pred - neg_pred)))
        _emb_loss = tf.nn.l2_loss(
            pu) + tf.nn.l2_loss(pi) + tf.nn.l2_loss(nu) + tf.nn.l2_loss(ni)
        return _mf_loss, _emb_loss

    def pred(self, sess, feed_dict):
        return sess.run(self.pred_op,
                        feed_dict=feed_dict)

    # @property
    def batch_pred(self, u, i):


        M_tmp = tf.reshape(tf.matmul(tf.reshape(u, [-1, self.emb_dim * (self.depth + 1)]), self.A),
                           [-1, self.num_mp_1, self.emb_dim * (self.depth + 1)])
        # ========================== user * pos_item
        # [bs, num_mp_1, d] * [bs, num_mp_2, d].transpose([0,2,1])
        #  = [bs, num_mp_1, num_mp_2]
        M_pos = tf.matmul(M_tmp, i, transpose_b=True)
        # get att via pool
        U_mp_att_pos = tf.nn.softmax(
            tf.reduce_max(M_pos, axis=2))  # [bs, num_mp_1]
        I_mp_att_pos = tf.nn.softmax(
            tf.reduce_max(M_pos, axis=1))  # [bs, num_mp_2]
        # att_embed_1 [bs, dim]
        att_embed_1 = tf.squeeze(tf.matmul(tf.transpose(u, [0, 2, 1]),  # [bs, dim, num_mp_1]
                                           tf.expand_dims(U_mp_att_pos, -1)))  # [bs, num_mp_1, 1]
        # att_embed_2 [bs, dim]
        att_embed_2 = tf.squeeze(tf.matmul(tf.transpose(i, [0, 2, 1]),  # [bs, dim, num_mp_1]
                                           tf.expand_dims(I_mp_att_pos, -1)))
        batch_rating = tf.matmul(att_embed_1, att_embed_2, transpose_b=True)
        return batch_rating

    def co_attention(self, transform=False):


        if transform:
            self.u_g_emb = tf.layers.dense(self.u_g_emb, self.emb_dim * (self.depth + 1),
                                           activation=tf.nn.elu, name='user_proj_co_attention')
            # pos and neg item use the same dense layer
            self.pos_i_g_emb = tf.layers.dense(self.pos_i_g_emb, self.emb_dim * (self.depth + 1),
                                               activation=tf.nn.elu, name='item_proj_co_attention')
            self.neg_i_g_emb = tf.layers.dense(self.neg_i_g_emb, self.emb_dim * (self.depth + 1),
                                               activation=tf.nn.elu, name='item_proj_co_attention',
                                               reuse=True)  #
        M_tmp = tf.reshape(tf.matmul(tf.reshape(self.u_g_emb, [-1, self.emb_dim * (self.depth + 1)]),
                                     self.A),
                           [-1, self.num_mp_1, self.emb_dim * (self.depth + 1)])
        # ========================== user * pos_item ==============
        # [bs, num_mp_1, d] * [bs, num_mp_2, d].transpose([0,2,1])
        #  = [bs, num_mp_1, num_mp_2]
        M_pos = tf.matmul(M_tmp, self.pos_i_g_emb, transpose_b=True)
        # get att via pool
        U_mp_att_pos = tf.nn.softmax(
            tf.reduce_max(M_pos, axis=2))  # [bs, num_mp_1]
        I_mp_att_pos = tf.nn.softmax(
            tf.reduce_max(M_pos, axis=1))  # [bs, num_mp_2]
        # att_embed_1 [bs, dim]
        att_embed_1 = tf.squeeze(tf.matmul(tf.transpose(self.u_g_emb, [0, 2, 1]),  # [bs, dim, num_mp_1]
                                           tf.expand_dims(U_mp_att_pos, -1)))  # [bs, num_mp_1, 1]
        # att_embed_2 [bs, dim]
        att_embed_2 = tf.squeeze(tf.matmul(tf.transpose(self.pos_i_g_emb, [0, 2, 1]),  # [bs, dim, num_mp_1]
                                           tf.expand_dims(I_mp_att_pos, -1)))  # [bs, num_mp_1, 1]
        # ========================== user * neg_item ===================
        # [bs, num_mp_1, d] * [bs, num_mp_2, d].transpose([0,2,1])
        #  = [bs, num_mp_1, num_mp_2]
        M_neg = tf.matmul(M_tmp, self.neg_i_g_emb, transpose_b=True)
        # get att via pool
        U_mp_att_neg = tf.nn.softmax(
            tf.reduce_max(M_neg, axis=2))  # [bs, num_mp_1]
        I_mp_att_neg = tf.nn.softmax(
            tf.reduce_max(M_neg, axis=1))  # [bs, num_mp_2]
        # att_embed_3 [bs, dim]
        att_embed_3 = tf.squeeze(tf.matmul(tf.transpose(self.u_g_emb, [0, 2, 1]),  # [bs, dim, num_mp_1]
                                           tf.expand_dims(U_mp_att_neg, -1)))  # [bs, num_mp_1, 1]
        # att_embed_4 [bs, dim]
        att_embed_4 = tf.squeeze(tf.matmul(tf.transpose(self.neg_i_g_emb, [0, 2, 1]),  # [bs, dim, num_mp_1]
                                           tf.expand_dims(I_mp_att_neg, -1)))  # [bs, num_mp_1, 1]
        return att_embed_1, att_embed_2, att_embed_3, att_embed_4



    def build_vars(self):
        # embedding & variables in all parts of our model
        initializer = tf.contrib.layers.xavier_initializer()
        if self.have_feature is None:
            print('init user/item embedding via xavier initialization')
            self.init_user_embed = tf.Variable(initializer(
                [self.n_users, self.num_dim_1]), name='init_user_embedding')
            self.init_item_embed = tf.Variable(initializer(
                [self.n_items, self.num_dim_2]), name='init_item_embedding')
        else:
            print('init user/item embedding via loading their features ')
            self.init_user_embed = tf.get_variable(name='init_user_embedding',
                                                   shape=[self.n_users, self.num_dim_1],
                                                   initializer=tf.constant_initializer(self.have_feature[0]),
                                                   trainable=False)
            self.init_item_embed = tf.get_variable(name='init_item_embedding',
                                                   shape=[self.n_items, self.num_dim_2],
                                                   initializer=tf.constant_initializer(self.have_feature[1]),
                                                   trainable=False)
        print(
            f"init user: {self.init_user_embed.get_shape().as_list()}, init Item: {self.init_item_embed.get_shape().as_list()}")
        # gnn weights
        if self.gnn_type == 'simple':
            self.gcn_Ws = []
            for i in range(self.depth):
                self.gcn_Ws.append(tf.Variable(initializer(
                    [self.emb_dim, self.emb_dim]), name=f"gcn_Ws_{i}"))
        # att mat
        self.A = tf.Variable(initializer(
            [self.emb_dim * (self.depth + 1), self.emb_dim * (self.depth + 1)]), name='attentive_matrix')

    def dual_gnn_update_embed(self):
        if self.U_adj_list is not None:
            mp_user_embed_list = self.gnn_update_embed(
                self.init_user_embed, self.U_adj_list, self.u_gnn_depth)
        else:
            mp_user_embed_list = []
        if self.I_adj_list is not None:
            mp_item_embed_list = self.gnn_update_embed(
                self.init_item_embed, self.I_adj_list, self.i_gnn_depth)
        else:
            mp_item_embed_list = []
        # import pdb;pdb.set_trace()
        if self.if_UIUI:
            U_embed_via_UI, I_embed_via_UI = self.UI_gnn_update(self.init_user_embed,
                                                                self.init_item_embed,
                                                                self.UIUI,
                                                                self.UIUI_depth)
            mp_user_embed_list.append(U_embed_via_UI)
            mp_item_embed_list.append(I_embed_via_UI)
        # import pdb;
        # pdb.set_trace()
        self.multi_user_embed = tf.concat(mp_user_embed_list, axis=1)
        self.multi_item_embed = tf.concat(mp_item_embed_list, axis=1)
        print(f"User embedding: {self.multi_user_embed.shape}, Item embedding: {self.multi_item_embed}")

    def UI_gnn_update(self, user_embed, item_embed, UIUI, depth):
        ui_embed = tf.concat([user_embed, item_embed], axis=0)
        if self.gnn_type == 'gat':
            ui_gnn_embed = self.gat(UIUI, ui_embed, depth)
        if self.gnn_type == 'simple':
            ui_gnn_embed = self.simple(UIUI, ui_embed, depth)
        if self.gnn_type == 'ngcf':
            ui_gnn_embed = self.ngcf(UIUI, ui_embed, depth)
        u_gnn_via_ui, i_gnn_via_ui = tf.split(ui_gnn_embed, [self.n_users, self.n_items], axis=0)
        return tf.expand_dims(u_gnn_via_ui, axis=1), tf.expand_dims(i_gnn_via_ui, axis=1)

    def gnn_update_embed(self, fea, adj_list, depth_list):
        """
        GAT for node and one meta-path
        fea: [N, d]
        adj: [N, N]
        for node and adj_list, return node embeding, [num_node, num_mp, emb_dim]
        TODO jk-net return node embeding, [num_node, num_mp, emb_dim*depth]
        """

        embed_list = []
        # updata based on U/M specific meta-path (e.g., M*M or U*U)
        # import pdb;
        # pdb.set_trace()
        for idx, adj in enumerate(adj_list):
            if self.gnn_type == 'simple':
                tmp = self.simple(adj, fea, depth_list[idx])
            if self.gnn_type == 'gat':
                tmp = self.gat(adj, fea, depth_list[idx])
            if self.gnn_type == 'ngcf':
                tmp = self.ngcf(adj, fea, depth_list[idx])
            # comb
            tmp = tf.expand_dims(tmp, axis=1)
            embed_list.append(tmp)
        # updata based on U-M, learn embed for both U and M

        # multi_embed = tf.concat(embed_list, axis=1)  # [N, num_mp, d]
        # return multi_embed
        return embed_list

    def ngcf(self, adj, fea, depth):
        h = fea
        embed_list = [h]
        for i in range(depth):
            h = self.ngcf_layer(adj, h, self.coef_drop[i])
            embed_list.append(h)

        return tf.concat(embed_list, axis=1)

    def ngcf_layer(self, adj, fea, drop):
        ego_embeddings = fea
        side_embeddings = adj @ fea
        sum_embeddings = tf.layers.dense(fea, self.emb_dim, activation=tf.nn.leaky_relu)
        bi_embeddings = tf.multiply(ego_embeddings,
                                    side_embeddings)
        bi_embeddings = tf.layers.dense(bi_embeddings, self.emb_dim, activation=tf.nn.leaky_relu)
        ego_embeddings = sum_embeddings + bi_embeddings
        # message dropout. 加了这一行ngcf从37%涨到了43%..
        ego_embeddings = tf.nn.dropout(
            ego_embeddings, 1 - drop)
        return ego_embeddings

    def simple_0815(self, adj, fea, depth):
        tmp = tf.cast(fea, tf.float64)
        for i in range(depth):
            tmp = self.simple_layer(adj, tmp)
        return tmp

    def simple(self, adj, fea, depth):
        tmp = tf.nn.dropout(fea, 1 - self.ffd_drop[0])
        tmp_list = [tmp]
        for i in range(depth):
            tmp = adj @ tmp @ self.gcn_Ws[i]
            tmp_list.append(tmp)
        tmp_all = tf.concat(tmp_list, axis=1)
        return tmp_all  # adj @ fea @ self.gcn_W

    def simple_layer(self, adj, fea, act_fn=tf.nn.elu):
        fea_proj = tf.layers.dense(fea, fea.shape[1], activation=None, use_bias=False)
        return tf.matmul(adj, fea_proj)

    def gcn(self, adj, fea):
        pass

    def gcn_layer(self, adj, fea):
        pass

    def gat_old(self, adj, fea, depth):
        tmp = fea
        for i in range(depth):
            tmp = self.gat_layer(adj, tmp)
        return tmp

    def gat_layer_old(self, adj, fea, act_fn=tf.nn.elu):


        mask_mat = -1e9 * (1.0 - adj)
        fea = tf.expand_dims(fea, axis=0)  # [1, num_node, fea_dim]
        fea = tf.layers.conv1d(fea, self.emb_dim, 1, use_bias=False)
        f1 = tf.layers.conv1d(fea, 1, 1)
        f2 = tf.layers.conv1d(fea, 1, 1)
        logits = f1 + tf.transpose(f2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits + mask_mat), axis=-1)
        # update embed
        vals = tf.matmul(coefs, fea)
        ret = tf.contrib.layers.bias_add(vals)
        return act_fn(tf.squeeze(ret))




    def gat_(self, adj, fea, depth):
        emb_list = [fea]
        h = tf.expand_dims(fea, axis=0)
        bias_mat = -1e9 * (1.0 - adj)
        for i in range(depth):
            pass

    def gat(self, adj, fea, depth):
        # embed_list = []
        all_order_emb_list = [fea]
        fea = tf.expand_dims(fea, axis=0)  # [1, num_node, fea_dim]
        bias_mat = -1e9 * (1.0 - adj)
        #

        # 1-layer
        h = self.attn_head(fea, bias_mat)
        all_order_emb_list.append(tf.squeeze(h))
        for i in range(1, depth):
            h = self.attn_head(h, bias_mat,
                               in_drop=self.ffd_drop[i],
                               coef_drop=self.coef_drop[i])
            all_order_emb_list.append(tf.squeeze(h))
        h_ = tf.squeeze(h)
        # import pdb;
        # pdb.set_trace()
        return tf.concat(all_order_emb_list, axis=1)
        # return h_

    def attn_head(self, seq, bias_mat, activation=tf.nn.elu, in_drop=0.0, coef_drop=0.0, residual=False):
        seq_fts = tf.layers.conv1d(seq, self.emb_dim, 1, use_bias=False)
        f1 = tf.layers.conv1d(seq_fts, 1, 1)
        f2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f1 + tf.transpose(f2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)
        if residual:
            ret = ret + seq
        # import pdb;pdb.set_trace()
        res = tf.nn.dropout(ret, 1.0 - coef_drop)
        return activation(res)

    def attn_head0819(self, seq, bias_mat,
                      activation=tf.nn.elu, in_drop=0.0, coef_drop=0.0, residual=False):


        with tf.name_scope('my_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)

            seq_fts = tf.layers.conv1d(seq, self.emb_dim, 1, use_bias=False)
            # import pdb;
            # pdb.set_trace()
            # simplest self-attention possible
            f_1 = tf.layers.conv1d(seq_fts, 1, 1)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1)
            logits = f_1 + tf.transpose(f_2, [0, 2, 1])
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            vals = tf.matmul(coefs, seq_fts)
            ret = tf.contrib.layers.bias_add(vals)

            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + tf.layers.conv1d(seq, ret.shape[-1], 1)  # activation
                else:
                    ret = ret + seq

            return activation(ret)  # activation



if __name__ == "__main__":
    # small test

    class config_ml_100k:
        path = './Data/ml-100k/'
        data_name = path.split('/')[-2]
        depth = 3
        only_UI = False  # True, model_only_UIUI; False, model_with_UIUI
        if_UIUI = True  # 是否基于UI来聚合同时得到U和I的表示
        gnn_type = 'ngcf'  # 'simple', 'gat', 'ngcf'
        # for model_with_UIUI
        co_attention_transform = True

        UIUI_depth = depth
        U_MP = ['UU']  # k表示基于相似性的过滤 UMU_k_10
        U_MP_name = "_".join(U_MP)
        U_Fea = 'UMG'
        I_MP = ['MGM']  # MUM_k_5
        I_MP_name = "_".join(I_MP)
        I_Fea = 'MG'
        # gnn depth of each mp for user/item

        u_gnn_depth = [depth] * len(U_MP)
        i_gnn_depth = [depth] * len(I_MP)
        max_degree = 10  # max neighbors of neighbors, resample or sample
        # gnn hyper para
        emb_dim = 64
        if gnn_type == 'ngcf':
            ffd_drop = [0.1] * depth
            coef_drop = [0.1] * depth
            # =========  hyper para
            lr = 1e-4
            l2_reg = 1e-5
        if gnn_type == 'gat':
            ffd_drop = [0.5] * depth
            coef_drop = [0.6] * depth
            # =========  hyper para
            lr = 1e-5
            l2_reg = 1e-5

        batch_size = 200
        #
        show_step = 1  # if 0, means do not show
        epoch = 800
        save_flag = True  # save model
        pretrain = False  # load trained model
        report_pretrain = True  # report performance after load pretrain
        # eval
        Ks = [10, 20, 40, 60, 80, 100]
        sample_negative = None  # 50 # in order to compare with MCRec, ref: eval.py -> test_one_user
        test_flag = False
        # ngcf hyper para
        model_type = 'ngcf'
        adj_type = 'norm'
        alg_type = 'ngcf'
        layer_size = [emb_dim] * depth
        embed_size = emb_dim
        verbose = 1
        gpu_id = 3


    model = HGRec_with_UI(cfg,
                          U_adj_list=[np.ones([10, 10], dtype=np.float32),
                                      np.ones([10, 10], dtype=np.float32)],
                          I_adj_list=[np.ones([20, 20], dtype=np.float32),
                                      np.ones([20, 20], dtype=np.float32),
                                      np.ones([20, 20], dtype=np.float32)],
                          UIUI=np.ones([10 + 20, 10 + 20])
                          )
