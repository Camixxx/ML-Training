# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name

import numpy as np
import mxnet as mx
from lstm import lstm_unroll
import read_music

def Perplexity(label, pred):
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

if __name__ == '__main__':
    batch_size = 10
    buckets = [10, 20, 30, 40, 50, 60]
    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 2

    input_size = 120
    num_output = 10
    invalid_label = 0

    num_epoch = 25
    learning_rate = 0.01
    momentum = 0.0

    # dummy data is used to test speed without IO
    dummy_data = False
    contexts = mx.cpu(0)
    #contexts = [mx.context.gpu(i) for i in range(1)]

    #vocab = default_build_vocab("./data/ptb.train.txt")

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len, input_size=input_size,
                           num_hidden=num_hidden, num_embed=num_output,
                           num_label=1)

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    train_sent = read_music.read_measure("601598.mid")
    val_sent = read_music.read_measure("601598.mid")

    # data_train = mx.io.NDArrayIter(train_sent)
    data_train = mx.rnn.BucketSentenceIter(train_sent, batch_size,
                                           invalid_label=invalid_label)

    data_val = mx.rnn.BucketSentenceIter(val_sent, batch_size,
                                         invalid_label=invalid_label)



    stack = lstm.lstm()
    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        label_class = mx.sym.Variable('class_label')
        embed = mx.sym.Embedding(data=data, input_dim=120,
                                 output_dim=num_embed, name='embed')
        stack.reset()
        outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)
        print("args:", outputs.list_arguments())
        pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))

        pred = mx.sym.FullyConnected(data=pred, num_hidden=num_hidden, name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    if len(buckets) == 1:
        # only 1 bucket, disable bucketing
        symbol = sym_gen(buckets[0])
    else:
        symbol = sym_gen

    model = mx.mod.BucketingModule(
        sym_gen=symbol,
        default_bucket_key=data_train.default_bucket_key,
        context=contexts)
    # import logging
    # head = '%(asctime)-15s %(message)s'
    # logging.basicConfig(level=logging.DEBUG, format=head)
    num_epochs = 12
    disp_batches = 50
    optimizer = "sgd"
    model.fit(    train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = ['accuracy'],
        optimizer           = optimizer,
        optimizer_params    = { 'learning_rate': 0.01,
                                'momentum': 0.0,
                                'wd': 0.00001},
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch           = num_epochs,
        batch_end_callback  = mx.callback.Speedometer(batch_size, disp_batches))

    out = model.get_outputs()[0].asnumpy();
    print("output:",out)
    print("end")

