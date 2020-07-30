

class config_ml_100k:
    path = './Data/ml-100k/'
    data_name = path.split('/')[-2]
    # time
    today = '0829_add_norm_larger_reg'
    # gnn based para
    gnn_type = 'ngcf'  # 'simple', 'gat', 'ngcf'
    restart_prob = 0.0  # APPNP model
    U_MP = ['UM','UU']  # k表示基于相似性的过滤 UMU_k_10
    U_MP_name = "_".join(U_MP)
    U_Fea = 'UMG'
    I_MP = ['MU']  # ['no']  # ['MGM']  # MUM_k_5
    I_MP_name = "_".join(I_MP)
    I_Fea = 'MG'
    depth = 3
    max_degree = 10
    u_gnn_depth = [depth] * len(U_MP)
    i_gnn_depth = [depth] * len(I_MP)
    emb_dim = 64
    if gnn_type == 'ngcf':
        ffd_drop = [0.8] * depth
        coef_drop = [0.8] * depth
        lr = 5e-4  # 1e-4
        l2_reg = 1e-2
    if gnn_type == 'gat':
        ffd_drop = [0.5] * depth
        coef_drop = [0.8] * depth
        lr = 1e-5
        l2_reg = 1e-5
    interaction_type = 'han'  # 'co_attention', 'han', 'concat', 'avg'
    co_attention_transform = False
    co_attention_type = 'max'  # 'mean'  # 'max'
    # initial embed via fea
    use_fea = False
    # initial embed via averged fea embed
    user_fea_embed = False
    # basic para
    batch_size = 800
    show_step = 1  # if 0, means do not show
    epoch = 800
    # save para
    save_flag = True  # save model
    pretrain = True  # load trained model
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

