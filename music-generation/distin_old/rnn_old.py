import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math


RNNState = namedtuple("RNNState", ["h"])
RNNParam = namedtuple("RNNParam", ["i2h_weight", "i2h_bias",
                                   "h2h_weight", "h2h_bias"])
RNNModel = namedtuple("RNNModel", ["rnn_exec", "symbol",
                                   "init_states", "last_states",
                                   "seq_data", "seq_labels", "seq_outputs",
                                   "param_blocks"])


def rnn(num_hidden, in_data, prev_state, param, seqidx, layeridx, dropout=0., batch_norm=False):
    if dropout > 0. :
        in_data = mx.sym.Dropout(data=in_data, p=dropout)

    i2h = mx.sym.FullyConnected(data=in_data,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    hidden = i2h + h2h
    hidden = mx.sym.Activation(data=hidden, act_type="tanh")

    if batch_norm == True:
        hidden = mx.sym.BatchNorm(data=hidden)

    return RNNState(h=hidden)

'''展开RNNs cell
num_rnn_layer:层数
seq_len:应该对应rnn中的时刻的长度，即展开的rnn总共的层数，每一层对应一个时刻
num_hidden:隐藏层节点数
num_label：每一行只有一个数据，则label维度为1
batch_norm: batch nomalization
'''
def rnn_unroll(num_rnn_layer, seq_len, num_hidden, num_embed, num_label, dropout=0., batch_norm=False):

    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    # param_cells是两层rnn中的weigh和bias，他们之间是使用fullconnected连接的
    # last_states是h状态，h是前一层rnn的输出
    # 这里是对每个rnn cell中的两层状态进行设置
    for i in range(num_rnn_layer):
        param_cells.append(RNNParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                    i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                    h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                    h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = RNNState(h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_rnn_layer)

    #label = mx.sym.Variable("label")
    last_hidden = []
    loss_all = []
    # last_hidden存储上一次hidden
    # loss_all存储损失
    # 这里对展开的每个rnn进行更新
    for seqidx in range(seq_len):
        data= mx.sym.Variable("data/%d" % seqidx)
        # stack RNN 对每层进行更新
        for i in range(num_rnn_layer):
            if i==0:
                dp=0.
            else:
                dp = dropout
            # 下一个状态
            next_state = rnn(num_hidden, in_data=data,
                             prev_state=last_states[i],
                             param=param_cells[i],
                             seqidx=seqidx, layeridx=i,
                             dropout=dp, batch_norm=batch_norm)
            # last_state中的h都是存储的是经过2层rnn处理之后的结果，每个隐藏单元的输出都是一个数值，现在进行更新
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        last_hidden.append(hidden)

        fc = mx.sym.FullyConnected(data=hidden, weight=cls_weight, bias=cls_bias,
                                   num_hidden=num_hidden)

        sm = mx.sym.SoftmaxOutput(data=fc, label=mx.sym.Variable("label/%d" % seqidx),
                                  name='sm')
        loss_all.append(sm)
        print("loss all")
        print(loss_all)
    # 返回两种写法，第一种是在循环内，data=hidden。然后用loss加起来
    # 第二种就是现在这样，用concat，写错了已经删去
    return mx.sym.Group(loss_all)

def is_param_name(name):
    return name.endswith("weight") or name.endswith("bias") or\
        name.endswith("gamma") or name.endswith("beta")

''' ctx:使用cpu还是gpu，
num_rnn_layer,seq_len,num_hidden,num_label 为对应rnn参数
batch_size：一个batch的数量
initializer：初始化
'''
def setup_rnn_model(ctx,
                    num_rnn_layer, seq_len, # seq_len=8, num_layer=2
                    num_hidden, num_label,  # num_hidden=200, num_embed=128
                    batch_size,   # batch_size=1
                    num_embed,
                    initializer,
                    dropout=0.):

    # 因为一行只有一个小节，所以num_label=1
    print('num_label',num_label)

    """set up rnn model with rnn cells，调用rnn展开，参数为：
    神经元层数、rnn层数、隐藏神经元数、label数、dropout
    """
    rnn_sym = rnn_unroll(num_rnn_layer=num_rnn_layer, seq_len=seq_len,num_embed=num_embed,
                          num_hidden=num_hidden,num_label=num_label,dropout=dropout)

    arg_names = rnn_sym.list_arguments()
    output_names = rnn_sym.list_outputs()
    print("____________arg_names")
    print(arg_names)
    print(len(arg_names))
    print("____________output_names")
    print(output_names)
    print(len(output_names))

    input_shapes = {}
    for name in arg_names:
        if name.endswith("init_h"):
            # 这里制定了第一时刻时，需要需要以来的prev_state, 设定是一个seq_len*num_hidden的矩阵
            input_shapes[name] = (batch_size, num_hidden)
        elif name.startswith("data"):
            input_shapes[name] = (batch_size, )
        else:
            pass

    # input_shapes除了有seq_num个输入数据，还0_init_h

    print ('input_shapes')
    print(input_shapes)
    '''arg_shape 和 arg_names是对应的，其中arg_names以list的形式存储rnn中的
    所有参数; arg_shape则以list存储对应arg_names参数的维度; arg_array也和
    arg_shape 和　arg_names对应，它也是list类型，其中每个元素是mx.nd，是
    每个参数的的存储空间'''
    # 在python中函数定义中传入一个星号的将会把传入的参数转换成tuple，而
    # 传入两个星号的表明将传入的参数转化成字典比如a=1,b=2 转化成{'a':1, 'b':2}
    arg_shape, out_shape, aux_shape = rnn_sym.infer_shape(**input_shapes)

    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]

    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        if is_param_name(name):
            args_grad[name] = mx.nd.zeros(shape, ctx)

    rnn_exec = rnn_sym.bind(ctx=ctx, args=arg_arrays,
                            args_grad=args_grad,
                            grad_req="add")
    param_blocks = []
    arg_dict = dict(zip(arg_names, rnn_exec.arg_arrays))
    print('arg_dict')
    print(arg_dict)
    for i, name in enumerate(arg_names):
        if is_param_name(name):
            initializer(name, arg_dict[name])

            param_blocks.append((i, arg_dict[name], args_grad[name], name))
        else:
            assert name not in args_grad

    for i in range(len(param_blocks)):
        print(param_blocks[i][0], param_blocks[i][3])

    out_dict = dict(zip(rnn_sym.list_outputs(), rnn_exec.outputs))

    init_states = [RNNState(h=arg_dict["l%d_init_h" % 1])]
    print('init_sates')
    print(init_states)
    print('rnn_exec.')
    print(rnn_exec)
    seq_labels = [rnn_exec.arg_dict["label/%d" % i] for i in range(seq_len)]
    seq_data = [rnn_exec.arg_dict["data/%d" % i] for i in range(seq_len)]

    last_states = [RNNState( h=out_dict["sm_output"]) for i in range(num_rnn_layer)]
    seq_outputs = out_dict["sm_output"]

    return RNNModel(rnn_exec=rnn_exec, symbol=rnn_sym,
                     init_states=init_states, last_states=last_states,
                     seq_data=seq_data, seq_labels=seq_labels, seq_outputs=seq_outputs,
                     param_blocks=param_blocks)


def set_rnn_inputs(m, X, begin):
    '''
    这里是为epoch即一个seq_num=8设置输入输出，输入都是一个128维的向量即为
    X的列数（即X的一行），输出为输入所在行的下一行，也是128维的向量.一行数据
    既作为该轮（seqidx)的输入，同时又作为上一轮(seqidx - 1)的输出。这样一个
    epoch就会又35 X 20 = 700的输出向量(seq_labels)
    '''
    seq_len = len(m.seq_data)
    batch_size = m.seq_data[0].shape[0]
    for seqidx in range(seq_len):
        idx = (begin + seqidx) % X.shape[0]
        next_idx = (begin + seqidx + 1) % X.shape[0]
        x = X[idx, :]
        y = X[next_idx, :]
       # mx.nd.array(x).copyto(m.seq_data[seqidx])
        m.seq_labels[seqidx * batch_size: seqidx * batch_size + batch_size] = y



def calc_nll(seq_label_probs, X, begin):
    nll = -np.sum(np.log(seq_label_probs) / len(X[0, :]))
    return nll


# max_grad_norm=5.0 | update_period=1 | wd=0 | learning_rate=0.1 | num_roud=25
def train_rnn(model, X_train_batch, X_val_batch,label,
               num_round, update_period,
               optimizer='sgd', half_life=2, max_grad_norm=5.0, **kwargs):
    print("Training swith train.shape=%s" % str(X_train_batch.shape))
    print("Training swith val.shape=%s" % str(X_val_batch.shape))
    print( "first row data：")
    print( X_train_batch[0])
    m = model
    seq_len = len(m.seq_data)
    batch_size = m.seq_data[0].shape[0]
    print("batch_size=%d" % batch_size)
    print("seq_len=%d" % seq_len)
    print("num_round=%d" % num_round)

    opt = mx.optimizer.create(optimizer, **kwargs)
    #print('opt:')
    #print (type(opt)) <class 'mxnet.optimizer.SGD'>

    updater = mx.optimizer.get_updater(opt)
    print('updater')
    print (type(updater))

    epoch_counter = 0
    log_period = max(1000 / seq_len, 1)
    last_perp = 10000000.0

    for iteration in range(10):
        nbatch = 0
        train_nll = 0
        # reset states
        for state in m.init_states:
          #  state.c[:] = 0.0
            state.h[:] = 0.0
        tic = time.time()
        print("X_train_batch.shape")
        print(X_train_batch.shape)
        assert X_train_batch.shape[0] % seq_len == 0
        assert X_val_batch.shape[0] % seq_len == 0

        seq_label_probs = 1

        for begin in range(0, X_train_batch.shape[0], seq_len):
            set_rnn_inputs(m, X_train_batch, begin=begin)
            m.rnn_exec.forward(is_train=True)
            print('m.seq_outputs')
            print(type(m.seq_outputs))
            print((m.seq_outputs.shape))
            print(m.seq_outputs.asnumpy())
            print('m.seq_labes')
            print(np.asarray(m.seq_labels))
            #print(m.seq_labels.shape)

            '''mx.nd.choos_element_0index这个的第一个参数是一个矩阵，在这里是一个2维
            的；第二个参数是一个一维向量，向量的长度和第一个参数的行数是一样长的，
            向量中的每一个值都是对应行的index。函数的功能是第二个参数的下标为第一个
            参数的行的索引下标，第二个参数的下标对应的值为第一个参数在该行的列索引下
            标,这样就会取出一个向量'''

            m.rnn_exec.backward()
            # transfer the states
            # 将前面的seq_num(35)个计算的到的last_state，作为下一个时刻(seq_num
            # =35)的输入，这样就相当于整个rnn的展开层是无限增加的
            for init, last in zip(m.init_states, m.last_states):
                # last.c.copyto(init.c)
                last.h.copyto(init.h)
            # update epoch counter
            epoch_counter += 1
            if epoch_counter % update_period == 0:
                # updare parameters
                norm = 0.
                for idx, weight, grad, name in m.param_blocks:
                    grad /= batch_size
                    l2_norm = mx.nd.norm(grad).asscalar()
                    norm += l2_norm * l2_norm
                norm = math.sqrt(norm)
                print('norm')
                print(norm)
                for idx, weight, grad, name in m.param_blocks:
                    if norm > max_grad_norm:
                        grad *= (max_grad_norm / norm)
                    updater(idx, grad, weight)
                    # reset gradient to zero
                    grad[:] = 0.0
            seq_label_probs = label
            train_nll += calc_nll(seq_label_probs, X_train_batch, begin=begin)

            nbatch = begin + seq_len
            if epoch_counter % log_period == 0:
                print("Epoch [%d] Train: NLL=%.3f, Perp=%.3f" % (
                    epoch_counter, train_nll / nbatch, np.exp(train_nll / nbatch)))
        # end of training loop
        toc = time.time()
        print("Iter [%d] Train: Time: %.3f sec, NLL=%.3f, Perp=%.3f" % (
            iteration, toc - tic, train_nll / nbatch, np.exp(train_nll / nbatch)))


    # 保存
    print("model save")
    # save model
    save_model_prefix = "rnn_sym"
    #if save_model_prefix is None:
     #   save_model_prefix = "rnn_sym"
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)
    print("checkpoint")
    print(checkpoint)
    if num_round == 25:
        print("Valid:")
        m.rnn_exec.forward(is_train=False)
        print('m.seq_outputs')
        print(type(m.seq_outputs))
        print((m.seq_outputs.shape))
        print(m.seq_outputs.asnumpy())
        print('m.seq_labes')
        print(np.asarray(m.seq_labels))
