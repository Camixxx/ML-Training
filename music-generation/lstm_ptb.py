# pylint:skip-file
#encoding:utf-8
import lstm
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import read_xml as read

"""
PennTreeBank Language Model
We would like to thanks Wojciech Zaremba for his Torch LSTM code

The data file can be found at:
https://github.com/dmlc/web-data/tree/master/mxnet/ptb
"""
# 这里我们加了一个token变量，用来测试train/valid使用的dic，本程序必须保证valid中的
# 出现的所有的word都必须出现在train数据
def load_data(path, token, dic=None,):
    fi = open(path)
    content = fi.read()
    # print type(content)
    content = content.replace('\n', '<eos>')
    # print type(content)
    # 前两部操作content都是str，现在content为list
    content = content.split(' ')
    # print type(content)
    print("Loading %s, size of data = %d" % (path, len(content)))
    # 生成一个和content一样长的x，x为一个list，每一个元素都是一个float，初始值0.0
    x = np.zeros(len(content))
    if dic == None:
        dic = {}
        # 载入验证数据时dic是用train生成的dic，保证字典统一
        # print >> sys.stdout, 'test valid'
    idx = 0
    # for循环实现两个功能，1.使用dic对content总的word进行编号;2.将content借助dic
    # 进行转化，其中下标x[i]，i和content一一对应word的位置，而x[i]的值代表的是word
    # 在dic中的编号
    for i in range(len(content)):
        word = content[i]
        if len(word) == 0:
            continue
        if not word in dic:
            dic[word] = idx
            idx += 1
            # 测试dic的编号
            if token == 'valid':
                print (sys.stdout, '%s' % word)
        x[i] = dic[word]
    print("Unique token: %d" % len(dic))
    return x, dic

# def drop_tail(X, seq_len):
#     shape = X.shape
#     # 得到的行数x.shape[0]在这个函数中还有进行按seq_len进行取模
#     # 使返回的X的行数可以被seq_len整除
#     # print >> sys.stdout, 'drop tail:shape[0]: %d' % shape[0]
#     nstep = int(shape[0] / seq_len)
#     # print 'ddd'
#     # print X[0:(nstep * seq_len), :].shape[0]
#     # print nstep
#     return X[0:(nstep * seq_len), :]

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
    num_hidden = 200
    num_embed = 128
    num_lstm_layer = 2

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
    # x_train= replicate_data(t_data, batch_size,num_embed,seq_len)
    # x_val = replicate_data(v_data, batch_size,num_embed,seq_len)

    x_train = np.asarray(t_data)
    x_val = np.asarray(v_data)
    print("after drop tail")
    print(x_train.shape)

    model = lstm.setup_rnn_model(mx.cpu(),
                                 num_lstm_layer=num_lstm_layer,
                                 seq_len=seq_len,
                                 num_hidden=num_hidden,
                                 num_embed=num_embed,
                                 num_label=1,
                                 batch_size=batch_size,
                                 input_size=input_size,
                                 initializer=mx.initializer.Uniform(0.1),dropout=0.5)
    # max_grad_norm=5.0 | update_period=1 | wd=0 | learning_rate=0.1 | num_roud=25
    lstm.train_lstm(model, x_train, x_val,
                    num_round=num_round,
                    half_life=2,
                    max_grad_norm = max_grad_norm,
                    update_period=update_period,
                    learning_rate=learning_rate,
                    wd=wd)
    #               momentum=momentum)

main()
