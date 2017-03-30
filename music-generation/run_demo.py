#########################################################
#
#         run_demo.py 测试
#
#########################################################
import sys
import mxnet as mx
from music.read_xml import *
from music.rnn_train import *


def Perplexity(label, pred):
    # TODO(tofix): we make a transpose of label here, because when
    # using the RNN cell, we called swap axis to the data.
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)


def test():
    train_file = ['601598.xml','compromise_0.xml','Faded_part2.xml','random']
    train_data = loadAll(train_file)

    valid_file = ['compromise_1.xml','random','Newt_Says_Goodbye_to_Tina_part3_tempo120.xml']
    valid_data = loadAll(valid_file)


    return True




# Load the iterator
def get_iterator(int_state):
    train_file = ['601598.xml']
    t_data, t_lable = loadAll(train_file)

    valid_file = ['compromise_1.xml']
    v_data, v_lable = loadAll(valid_file)

    train = mx.io.MNISTIter(
        data=t_data,
        label=t_lable,
        shuffle=True)

    val = mx.io.MNISTIter(
        data=v_data,
        label=v_lable)

    return train, val


if __name__ == '__main__':
    batch_size = 32
    input_size = 16 * 8
    #buckets = [10, 20, 30, 40, 50, 60]
    #buckets = [32]
    buckets = []
    num_hidden = 200
    num_embed = 200
    num_rnn_layer = 2

    num_epoch = 1
    learning_rate = 0.01
    momentum = 0.0

    # dummy data is used to test speed without IO
    dummy_data = False

    # train
    devs = mx.cpu()

    #contexts = [mx.context.gpu(i) for i in range(1)]
    #vocab = default_build_vocab("./data/ptb.train.txt")

    def sym_gen(seq_len):
        return rnn_unroll(num_rnn_layer, seq_len, input_size,
                           num_hidden=num_hidden, num_embed=num_embed,
                           num_label=num_label)

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    #data_train = BucketSentenceIter("./data/ptb.train.txt", vocab,
      #                              buckets, batch_size, init_states)
    #data_val = BucketSentenceIter("./data/ptb.valid.txt", vocab,
     #                             buckets, batch_size, init_states)

    # if dummy_data:
    #     data_train = DummyIter(data_train)
    #     data_val = DummyIter(data_val)

    data_train,data_val= get_iterator(init_states)

    if len(buckets) == 1:
        # only 1 bucket, disable bucketing
        symbol = sym_gen(buckets[0])
    else:
        symbol = sym_gen

    model = mx.model.FeedForward(ctx=devs,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(data_train, eval_data=data_val, num_epoch=num_epoch,
        eval_metric=mx.metric.np(Perplexity),
        batch_end_callback=mx.callback.Speedometer(batch_size, 50),
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        optimizer='sgd',
        optimizer_params={'learning_rate': learning_rate,
                          'momentum': momentum, 'wd': 0.00001})

