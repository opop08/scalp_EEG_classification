import random
import tsai
from tsai.all import *
import numpy as np

c_in = 1
c_out = 3
seq_len = 256

train_param_grid = {
    'MLP': 150,
    'FCN': 200,
    'TCN': 80,
    'ResNet': 30,
    'xresnet1d50': 9,
    'InceptionTime': 25,
    'XceptionTime': 16,
    'MLSTM': 80,
    'mWDN': 35,
}

nn_param_grid = {
    'MLP': {
        'c_in': [1],
        'c_out': [3],
        'seq_len': [256],
        'layers': [[32] * 3, [64] * 3, [128] * 3, [256] * 3],
        'ps': [[0.1] * 3, [0.2] * 3, [0.3] * 3],
        'fc_dropout': [0.0, 0.05, 0.1, 0.2, 0.4]},

    'FCN': {
        'c_in': [1],
        'c_out': [3],
        # 'seq_len': [256],
        'layers': [[32] * 3, [64] * 3, [128] * 3, [256] * 3],
        'kss': [[7] * 3, [5] * 3, [3] * 3],
        # 'conv_dropout': [0.0, 0.05, 0.1, 0.2, 0.4],
        # 'fc_dropout': [0.0, 0.05, 0.1, 0.2, 0.4]
    },

    'TCN': {
        'c_in': [1],
        'c_out': [3],
        # 'seq_len': [256],
        'layers': [[16] * 7,
                   [32] * 7,
                   [64] * 7],
        'ks': [3, 5, 7, 9, 20, 30, 40],
        'conv_dropout': [0.0, 0.05, 0.1, 0.2, 0.4],
        'fc_dropout': [0.0, 0.05, 0.1, 0.2, 0.4]},

    'ResNet': {
        'c_in': [1],
        'c_out': [3],
        'seq_len': [256],
        # 'ks': [[3], [5], [7], [9], [20], [30], [40]],
    },

    'xresnet1d50': {
        'c_in': [1],
        'c_out': [3],
        # 'seq_len': [256],
        'ks': [3, 5, 7, 9, 15],
    },

    'InceptionTime': {
        'c_in': [1],
        'c_out': [3],
        # 'seq_len': [256],
        # 'ks': [3, 5, 7, 9, 20, 30, 40],
        'nf': [4, 8, 16],
        'depth': [3, 4, 5, 6]
    },

    'XceptionTime': {
        'c_in': [1],
        'c_out': [3],
        # 'seq_len': [256],
        'ks': [3, 5, 7, 9, 20, 30, 40],
        'nf': [4, 8, 16, 32],
        'adaptive_size': [20, 30, 40, 50]
    },

    'MLSTM': {
        'c_in': [1],
        'c_out': [3],
        'seq_len': [256],
        'conv_layers': [[32] * 3, [64] * 3, [128] * 3, [256] * 3],
        'kss': [[9] * 3, [7] * 3, [5] * 3, [3] * 3],
        'rnn_dropout': [0.1, 0.3, 0.5, 0.7, 0.9]
    },

    'mWDN': {
        'c_in': [1],
        'c_out': [3],
        'seq_len': [256],
        'ks': [3, 5, 7, 9],
        'nf': [8, 16, 32],
        'depth': [3, 6],
        'levels': [3, 6],
        'cell_dropout': [0.0, 0.05, 0.1, 0.2],
        'rnn_dropout': [0.3, 0.5, 0.7]
    },
}


def get_one_para_dict_random(model_name):
    if model_name not in nn_param_grid.keys():
        return {}

    model_dict = nn_param_grid[model_name]

    out_dict = {}
    for key in model_dict.keys():
        out_dict[key] = random.choice(model_dict[key])
    return out_dict


def get_key(one_dict):
    out_str = ''
    for k, v in one_dict.items():
        out_str += str(v)
    return out_str


def get_para_list(model_name):
    if model_name not in nn_param_grid.keys():
        return []

    out_list = []
    temp_dict = {}
    for i in range(0, 10000):
        one_dict = get_one_para_dict_random(model_name)
        cur_key = get_key(one_dict)
        temp_dict[cur_key] = one_dict

    for k, v in temp_dict.items():
        out_list.append(v)

    return out_list


X = np.load('./data/X.npy')
y = np.load('./data/y.npy')

if __name__ == '__main__':


    name_list = ["MLP", "FCN", "TCN", "InceptionTime", "ResNet",
                 "XceptionTime", "xresnet1d50", "MLSTM", "mWDN"]

    data_len = np.shape(y)[0]
    train_list = list(np.arange(0, int(data_len * 0.8), 1))
    val_list = list(np.arange(int(data_len * 0.8), data_len, 1))
    splits = tuple([train_list, val_list])

    tfms = [None, [Categorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=256 * 2, batch_tfms=[TSStandardize()], num_workers=0)


    for name in name_list[7:8]:

        para_list = get_para_list(name)

        for one_para_dict in para_list[:1]:

            if name == 'MLP':
                cur_model = MLP(**one_para_dict)
            if name == 'FCN':
                cur_model = FCN(**one_para_dict)
            if name == 'TCN':
                cur_model = TCN(**one_para_dict)
            if name == 'ResNet':
                cur_model = ResNetPlus(**one_para_dict)
            if name == 'xresnet1d50':
                cur_model = xresnet1d50_deeper(**one_para_dict)
            if name == 'InceptionTime':
                cur_model = InceptionTime(**one_para_dict)
            if name == 'XceptionTime':
                cur_model = XceptionTimePlus(**one_para_dict)
            if name == 'MLSTM':
                cur_model = MLSTM_FCNPlus(**one_para_dict)
            if name == 'mWDN':
                cur_model = mWDN(**one_para_dict)

            learn = Learner(dls, cur_model, metrics=accuracy)
            learn.fit_one_cycle(50, lr_max=3 * 1e-3)

    exit()
