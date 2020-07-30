#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
"""

"""

import multiprocessing
import os
import sys
from time import time

import tensorflow as tf

#
from model_old import HGRec
from model_simple import HGRec_simple
from model_with_UI_old_version import HGRec_UI
from model_only_UI import HGRec_only_UI
from model_with_UI import HGRec_with_UI
from model import HGRec

cores = multiprocessing.cpu_count() // 2
# Tensorflow

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# eval
from eval import *

from config import config_ml_100k as cfg

print('============================ config ============')
print(cfg.__dict__)


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (
            expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


# ===================    train   =======================
data_generator = Data_loader(cfg)

U_adj_list = data_generator.U_mp_list
# print(np.sum(U_adj_list[0]))
I_adj_list = data_generator.I_mp_list
U_fea = data_generator.U_fea
I_fea = data_generator.I_fea
UIUI = data_generator.UIUI
# Real data ml-100k
# model = HGRec(cfg,
#               U_adj_list, I_adj_list,
#               have_feature=[U_fea, I_fea]
#               )
t0 = time()
model = HGRec(cfg,
              U_adj_list,
              I_adj_list,
              UIUI,
              have_feature=[U_fea, I_fea]
              )
print("=========== model =========")
print(model)
saver = tf.train.Saver()
# *********************************************************
# save the model parameters.
if cfg.save_flag:
    # layer = '-'.join([str(l) for l in eval(args.layer_size)])
    # weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (
    #     args.proj_path, args.dataset, model.model_type, layer, str(args.lr),
    #     '-'.join([str(r) for r in eval(args.regs)]))
    weights_save_path = f"./saved_model_{cfg.today}/{cfg.data_name}/{cfg.U_MP_name}_{cfg.I_MP_name}_{cfg.gnn_type}_{cfg.interaction_type}_{cfg.depth}_{cfg.co_attention_type}"
    ensureDir(weights_save_path)
    save_saver = tf.train.Saver(max_to_keep=1)

sess = tf.Session(config=tf_config)
"""
*********************************************************
Reload the pretrained model parameters.ss
"""
if cfg.pretrain:
    # layer = '-'.join([str(l) for l in eval(args.layer_size)])

    pretrain_path = f"./saved_model_{cfg.today}/{cfg.data_name}/{cfg.U_MP_name}_{cfg.I_MP_name}_{cfg.gnn_type}_{cfg.interaction_type}_{cfg.depth}_{cfg.co_attention_type}"

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('load the pretrained model parameters from: ', pretrain_path)

        # *********************************************************
        # get the performance from pretrained model.
        if cfg.report_pretrain:
            users_to_test = list(data_generator.test_set.keys())
            ret = test(sess, model, users_to_test, drop_flag=True)
            cur_best_pre_0 = ret['recall'][0]

            pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                           'ndcg=[%.5f, %.5f], auc=[%.5f]' % \
                           (ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1],
                            ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1],ret['auc'])
            print(pretrain_ret)
    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining(not find save model).')

else:
    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.
    print('without pretraining (pretain flag = False).')

loss_loger = []
pre_loger = []
rec_loger = []
ndcg_loger = []
hit_loger = []
auc_loger=[]
stopping_step = 0
should_stop = False
final_attu=[]
final_atti=[]
best_ngcf = 0
for epoch in range(cfg.epoch):
    # print(f"start epoch {epoch}...")
    t1 = time()
    loss, base_loss, reg_loss = 0, 0, 0
    n_batch = int(data_generator.n_train // cfg.batch_size)
    for idx in range(n_batch):
        batch_data = data_generator.generate_train_batch(cfg.batch_size)
        feed_dict = {
            model.users: batch_data['users'],
            model.pos_items: batch_data['pos_items'],
            model.neg_items: batch_data['neg_items'],
            model.coef_drop: cfg.coef_drop,
            model.ffd_drop: cfg.ffd_drop
        }
        # data_generator.generate_train_feed_dict(model, batch_data)
        _, batch_loss, batch_base_loss, batch_reg_loss = model.train(sess,
                                                                     feed_dict=feed_dict)
        loss += batch_loss
        base_loss += batch_base_loss
        reg_loss += batch_reg_loss
    if cfg.interaction_type == 'han':
        u_att, i_att = model.get_han_att(sess, feed_dict=feed_dict)
        final_attu.append(np.mean(u_att))
        final_atti.append(np.mean(i_att))

        print(f"ATT: u_att: {np.mean(u_att, axis=0)}, i_att: {np.mean(i_att, axis=0)}")
    # import pdb; pdb.set_trace()
    if np.isnan(loss) == True:
        print('Loss is NAN !')
        sys.exit()
    # if epoch % cfg.show_step == 0:
    #     print(
    #         f"Epoch: {epoch}, Time: {time() - t1:.2f}, Loss: {loss:.2f}, base_loss: {base_loss:.2f}, reg_loss: {reg_loss:.2f}")
    # continue
    # evals s
    if epoch % 100 == 0:
        print(cfg.__dict__)

    if epoch % 20 == 0:
        # print(f"lr : {batch_lr}")

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)  # , sample_negative=cfg.sample_negative)

        t3 = time()
        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        auc_loger.append(ret['auc'])
        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], \nreca=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f], ' \
                   '\nprec=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f], \nhitt=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f], \nndcg=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f],\nauc=[%.5f]' % \
                   (epoch, t2 - t1, t3 - t2, loss, base_loss, reg_loss,
                    ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][3],
                    ret['recall'][4], ret['recall'][5],
                    ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][3],
                    ret['precision'][4], ret['precision'][5],
                    ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][3],
                    ret['hit_ratio'][4], ret['hit_ratio'][5],
                    ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][3], ret['ndcg'][4],
                    ret['ndcg'][5],ret['auc']
                    )
        print(perf_str)
        # print(model.lr)
        best_ngcf = max(best_ngcf, ret['ndcg'][0])
        if best_ngcf == ret['ndcg'][0] and cfg.save_flag:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    if epoch % 1 == 0:
        perf_str = 'Epoch %d : train==[%.5f=%.5f + %.5f]' % \
                   (epoch, loss, base_loss, reg_loss)
        print(perf_str)
    
    if ret['recall'][0] == cur_best_pre_0 and cfg.save_flag:
        save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
        print('save the weights in path: ', weights_save_path)

recs = np.array(rec_loger)
pres = np.array(pre_loger)
ndcgs = np.array(ndcg_loger)
hit = np.array(hit_loger)
auc=np.array(auc_loger)
best_rec_0 = max(recs[:, 0])
idx = list(recs[:, 0]).index(best_rec_0)

print('============================ config ============')
print(cfg.__dict__)
print(model)

final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
             (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
              '\t'.join(['%.5f' % r for r in pres[idx]]),
              '\t'.join(['%.5f' % r for r in hit[idx]]),
              '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
print(final_perf)
print('auc'+str(auc[idx]))
print(f"FINAL ATT: u_att: {np.mean(final_attu, axis=0)}, i_att: {np.mean(final_atti, axis=0)}")

save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
print('save the weights in path: ', weights_save_path)
