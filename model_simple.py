#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os

import numpy as np
import tensorflow as tf


class HGRec_simple:
    """



    """

    def __init__(self, cfg,
                 U_adj_list=None,  # only one mp [adj]
                 I_adj_list=None,  # only one mp [adj]
                 have_feature=None):
        print('Model: HGRec-simple (one metapath, no co-attention)')
        # have_features = [U_fea_mat, I_fea_mat]

        self.gnn_type = cfg.gnn_type
        self.u_gnn_depth = cfg.u_gnn_depth
        self.i_gnn_depth = cfg.i_gnn_depth
        # hyper para
        self.emb_dim = cfg.emb_dim
        self.l2_reg = cfg.l2_reg
        self.lr = cfg.lr
        self.max_degree = cfg.max_degree

        self.U_adj_list = U_adj_list
        self.I_adj_list = I_adj_list

        self.have_feature = have_feature
        # multi_embed_1.get_shape().as_list()[1]
        self.num_mp_1 = len(U_adj_list)
        # multi_embed_2.get_shape().as_list()[1]
        self.num_mp_2 = len(I_adj_list)
        if self.have_feature is None:
            self.num_dim_1 = self.emb_dim
            self.num_dim_2 = self.emb_dim
        else:
            self.num_dim_1 = have_feature[0].shape[1]
            self.num_dim_2 = have_feature[1].shape[1]

        self.n_users = U_adj_list[0].shape[0]
        self.n_items = I_adj_list[0].shape[0]

        self.holders = self.build_holders()
        # self.all_weights = self.init_vars()
        self.init_vars()
        self.dual_gnn_update_embed()
        # place hoders
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        # ======== 
        self.u_g_emb = tf.nn.embedding_lookup(
            self.multi_user_embed, self.users)
        self.pos_i_g_emb = tf.nn.embedding_lookup(self.multi_item_embed, self.pos_items)
        self.neg_i_g_emb = tf.nn.embedding_lookup(
            self.multi_item_embed, self.neg_items)
        # co attention
        # self.pos_u_att_emb, self.pos_i_att_emb, self.neg_u_att_emb, self.neg_i_att_emb = self.co_attention()
        # bpr_loss
        self.mf_loss, self.emb_loss = self.bpr(self.u_g_emb, self.pos_i_g_emb, self.u_g_emb,
                                               self.neg_i_g_emb)
        self.loss = self.mf_loss + self.l2_reg * self.emb_loss
        self.emb_loss_with_coef = self.l2_reg * self.emb_loss
        

        self.pred_op = tf.matmul(self.u_g_emb, self.pos_i_g_emb, transpose_b=True)
        # self.batch_pred(self.u_g_emb, self.pos_i_g_emb)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        # for NGCF
        # self.batch_ratings = self.pred_op

    def train(self, sess, feed_dict):
        return sess.run([self.opt, self.loss, self.mf_loss, self.emb_loss_with_coef],
                        feed_dict=feed_dict)

    def bpr(self, pu, pi, nu, ni):
      

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
        # import pdb; pdb.set_trace()
        # 把co-attention解耦了, 方便batch_pred
        M_tmp = tf.reshape(tf.matmul(tf.reshape(u, [-1, self.emb_dim]), self.A),
                           [-1, self.num_mp_1, self.emb_dim])
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

    def co_attention(self):


        M_tmp = tf.reshape(tf.matmul(tf.reshape(self.u_g_emb, [-1, self.emb_dim]), self.A),
                           [-1, self.num_mp_1, self.emb_dim])
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

    def build_holders(self):
        # build placeholders for user-item pair
        holders = {
            'user_input': tf.placeholder(tf.int32, [None, ]),
            'pos_item_input': tf.placeholder(tf.int32, [None, ]),
            'neg_item_input': tf.placeholder(tf.int32, [None, ])
        }
        return holders

    def init_vars(self):
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
        # gnn weights

        # att mat
        self.A = tf.Variable(initializer(
            [self.emb_dim, self.emb_dim]), name='attentive_matrix')

    def dual_gnn_update_embed(self):
        self.multi_user_embed = self.gnn_update_embed(
            self.init_user_embed, self.U_adj_list, self.u_gnn_depth)
        self.multi_item_embed = self.gnn_update_embed(
            self.init_item_embed, self.I_adj_list, self.i_gnn_depth)

    def gnn_update_embed(self, fea, adj_list, depth_list):
        """
        GAT for node and one meta-path
        fea: [N, d]
        adj: [N, N]
        for node and adj_list, return node embeding, [num_node, num_mp, emb_dim]
        TODO jk-net return node embeding, [num_node, num_mp, emb_dim*depth]
        """

        embed_list = []

        for idx, adj in enumerate(adj_list):
            if self.gnn_type == 'simple':
                tmp = self.simple(adj, fea, depth_list[idx])
            if self.gnn_type == 'gat':
                tmp = self.gat(adj, fea, depth_list[idx])
            # comb
            tmp = tf.expand_dims(tmp, axis=1)
            embed_list.append(tmp)

        multi_emebd = tf.concat(embed_list, axis=1)  # [N, num_mp, d]
        return multi_emebd

    def simple(self, adj, fea, depth):
        return tf.matmul(adj, fea)

    def gcn(self, adj, fea):
        pass

    def gcn_layer(self, adj, fea):
        pass

    def gat(self, adj, fea, depth):
        tmp = fea
        for i in range(depth):
            tmp = self.gat_layer(adj, tmp)
        return tmp

    def gat_layer(self, adj, fea, act_fn=tf.nn.elu):


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


# update embed for user/item like 19WWW_han
# U_MP_list = [U_MP_1_adj_mat, ..., ...]

# attentive matrix A
# A = tf.Variable(initializer([self.n_u_mp, self.n_i_mp]),
#                 name='attentive_matrix')


if __name__ == "__main__":
    # small test
    from config import config_ml_100k as cfg

    model = HGRec(cfg,
                  U_adj_list=[np.ones([10, 10], dtype=np.float32),
                              np.ones([10, 10], dtype=np.float32)],
                  I_adj_list=[np.ones([20, 20], dtype=np.float32), np.ones([20, 20], dtype=np.float32)],
                  have_feature=[np.ones([10, 128], dtype=np.float32),
                                np.ones([20, 128], dtype=np.float32)]
                  )
