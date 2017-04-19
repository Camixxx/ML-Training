import sys
import mxnet as mx
import numpy as np
import rnn
import read_xml as read
sys.path.insert(0, "../../python")
"""
For music classification

"""

def drop_tail(X, seq_len,num_size):
    shape = X.shape
    # 得到的行数x.shape[0] ,在这个函数中还有进行按seq_len进行取模
    # 使返回的X的行数可以被seq_len整除
    print(sys.stdout, 'drop tail:shape[0]: %d' % shape[0])
    nstep = int(shape[0] / seq_len)
    # print 'ddd'
    # print X[0:(nstep * seq_len), :].shape[0]
    # print nstep
    return X[0:(nstep * seq_len), :,num_size]

'''
转换为batch-size行，且每行数据都能被seq_len整除
并且转成np array，每个数据是一个128维向量
'''
def replicate_data(x, batch_size,num_size,seq_len):
    print(sys.stdout, 'batch_size: %d' % batch_size)
    print(sys.stdout, 'x.len: %d' % len(x))
    nbatch = int(len(x) / (batch_size* seq_len))
    #print(sys.stdout, 'nbatch: %d' % (nbatch))
    x_cut = np.asarray(x[:nbatch * batch_size])
    print(sys.stdout, 'x_cut size:' , x_cut)
    # 这里将x_cut转换成一个二维矩阵, 矩阵有nbatch行，batch_sie列
    data = x_cut.reshape(nbatch, batch_size,num_size)
    return data


def main():
    batch_size = 1
    seq_len = 8
    num_label = 1 # 因为只有一列数据
    num_hidden = 128
    num_size = 128
    num_rnn_layer = 2
    num_round = 25
    learning_rate= 0.1
    wd=0.
    momentum=0.0
    max_grad_norm = 5.0
    update_period = 1

    train_file = ['601598.xml']
    t_data, t_label = read.loadAll(train_file)
    valid_file = ['601598.xml']
    v_data, v_label = read.loadAll(valid_file)
    input_size = len(t_data)
    # print("Finish read xml, t_data.length =%d", input_size)
    # print("t_label")
    # print(t_label)
    #
    # print("before replicate")
    # print(len(t_data))
    #
    # x_train= replicate_data(t_data, batch_size,num_size,seq_len)
    # x_val = replicate_data(v_data, batch_size,num_size,seq_len)
    x_train = np.asarray(t_data)
    x_val = np.asarray(v_data)
    x_label = np.asarray(t_label)
    print("after drop tail")
    print(x_train.shape)

    print("after drop tail")
    print(x_train.shape)
    model = rnn.setup_rnn_model(mx.cpu(),
                                 num_rnn_layer=num_rnn_layer,
                                 seq_len=seq_len,
                                 num_hidden=num_hidden,
                                 num_label=num_label,
                                 batch_size=batch_size,
                                 num_embed = num_size,
                                 initializer=mx.initializer.Uniform(0.1),
                                 dropout=0)
    # max_grad_norm=5.0 | update_period=1 | wd=0 | learning_rate=0.1 | num_roud=25

    print("x train shape")
    print(x_train.shape)
    rnn.train_rnn(model, x_train, x_val, x_label,
                    num_round=num_round,
                    half_life=2,
                    max_grad_norm = max_grad_norm,
                    update_period=update_period,
                    learning_rate=learning_rate,
                    wd=wd)

main()