# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name

import numpy as np
import mxnet as mx
import read_music

def Perplexity(label, pred):
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)


if __name__ == '__main__':
    batch_size = 10
    buckets = [10, 20, 30, 40, 50, 60]
    num_hidden = 30
    num_embed = 200
    num_lstm_layer = 2

    input_size = 120
    num_output = 10
    invalid_label = 0

    num_epochs = 5
    disp_batches = 50
    optimizer = "sgd"
    lr = 0.01       #learning_rate
    wd = 0.00001
    mom = 0.0       #momentum

    # dummy data is used to test speed without IO
    dummy_data = False
    contexts = mx.cpu(0)
    default_bucket_key = 8
    # contexts = [mx.context.gpu(i) for i in range(1)]

    train_sent = read_music.read_measure("601598.mid")
    train_label = [1 for i in range(len(train_sent))]
    val_sent = read_music.read_measure("601598.mid")
    val_label = [1 for i in range(len(val_sent))]

    data_train = mx.io.NDArrayIter(mx.nd.array(train_sent), label=train_label)
    data_val = mx.io.NDArrayIter(mx.nd.array(val_sent), label=val_label)
    # data_train = mx.rnn.BucketSentenceIter(train_sent, batch_size,
    #                                        invalid_label=invalid_label)
    #
    # data_val = mx.rnn.BucketSentenceIter(val_sent, batch_size,
    #                                      invalid_label=invalid_label)

    rnn_cell = mx.rnn.BaseRNNCell("rnn")
    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('logistic_label')

        # embed = mx.sym.Embedding(data=data, input_dim=120,
        #                          output_dim=num_embed, name='embed')

        outputs, states = rnn_cell.unroll(length=seq_len, inputs=data, merge_outputs=True)
        '''inputs.shape(batch_size, length, ...) if layout == NTC'''
        print("args outputs:", outputs.list_arguments())
        print("args stats:", states.list_arguments())

        pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
        label = mx.sym.Reshape(label, shape=(-1,))

        pred = mx.sym.FullyConnected(data=pred, num_hidden= num_hidden, name='pred')
        pred = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='logistic')

        print("args pred:", pred.list_arguments())
        print("args label:", label.list_arguments())

        return pred, ('data',), ('logistic_label',)

    # if len(buckets) == 1:
    #     # only 1 bucket, disable bucketing
    #     symbol = sym_gen(buckets[0])
    # else:
    #     symbol = sym_gen

    # model = mx.mod.BucketingModule(
    #     sym_gen=sym_gen,
    #     default_bucket_key=data_train.default_bucket_key,
    #     context=contexts)

    model = mx.mod.BaseModule(sym_gen(12))
    # import logging
    # head = '%(asctime)-15s %(message)s'
    # logging.basicConfig(level=logging.DEBUG, format=head)

    model_prefix = "demo"
    # load model
    # if model_prefix is not None:
    #     model_prefix += "-%d"
    # model_args = {}
    # if args.load_epoch is not None:
    #     assert model_prefix is not None
    #     tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
    #     model_args = {'arg_params': tmp.arg_params,
    #                   'aux_params': tmp.aux_params,
    #                   'begin_epoch': args.load_epoch}
    #     # TODO: check epoch_size for 'dist_sync'
    #     epoch_size = args.num_examples / args.batch_size
    #     model_args['begin_num_update'] = epoch_size * args.load_epoch

    # save model
    #model_prefix = None
    checkpoint = None if model_prefix is None else mx.callback.do_checkpoint(model_prefix)

    model.fit(
        train_data=data_train,
        eval_data=data_val,
        eval_metric=mx.metric.Perplexity(1),
        # eval_metric         = ['accuracy'],
        optimizer= optimizer,
        optimizer_params={'learning_rate': lr,
                          'momentum': mom,
                          'wd': wd},
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch=num_epochs,
        batch_end_callback=mx.callback.Speedometer(batch_size, disp_batches),
        epoch_end_callback=checkpoint)

    out = model.get_outputs()[0].asnumpy()
    print("output shape:", out.shape)
    print(out)
    print("end")

