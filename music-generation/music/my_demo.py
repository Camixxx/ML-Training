import numpy as np
import mxnet as mx
import argparse,read_music

parser = argparse.ArgumentParser(description="Train RNN on Penn Tree Bank",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of stacked RNN layers')
parser.add_argument('--num-hidden', type=int, default=24,
                    help='hidden layer size')
parser.add_argument('--num-embed', type=int, default=120,
                    help='embedding layer size')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ' \
                         'Increase batch size when using multiple gpus for best performance.')
parser.add_argument('--kv-store', type=str, default='device',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=6,
                    help='max num of epochs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='the optimizer type')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--wd', type=float, default=0.00001,
                    help='weight decay for sgd')
parser.add_argument('--batch-size', type=int, default=1,
                    help='the batch size.')
parser.add_argument('--disp-batches', type=int, default=10,
                    help='show progress for every n batches')

output_dim = 10
input_dim = 8
def tokenize_text(fname, vocab=None, invalid_label=-1, start_label=0):
    lines = open(fname).readlines()
    lines = [filter(None, i.split(' ')) for i in lines]
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label,
                                               start_label=start_label)
    return sentences, vocab


if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    args = parser.parse_args()

    #buckets = []
    #buckets = [10, 20, 30, 40, 50, 60]
    buckets = [6, 9, 12, 15, 18, 21, 24]
    start_label = 1
    invalid_label = 0

    # train_sent, vocab = tokenize_text("./data/ptb.train.txt", start_label=start_label,
    #                                   invalid_label=invalid_label)
    # val_sent, _ = tokenize_text("./data/ptb.test.txt", vocab=vocab, start_label=start_label,
    #                             invalid_label=invalid_label)

    train_sent = read_music.read_measure("601598.mid")
    val_sent = read_music.read_measure("601598.mid")

    #data_train = mx.io.NDArrayIter(train_sent)
    data_train = mx.rnn.BucketSentenceIter(train_sent, args.batch_size,
                                             invalid_label=invalid_label)

    data_val = mx.rnn.BucketSentenceIter(val_sent, args.batch_size,
                                             invalid_label=invalid_label)
    w = data_train.getdata()
    stack = mx.rnn.SequentialRNNCell()
    for i in range(args.num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_l%d_'%i))

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        # embed = mx.sym.Embedding(data=data, input_dim=120,
        #                           output_dim=args.num_embed, name='embed')
        embed = mx.sym.Embedding(data=data, input_dim=input_dim,
                                 output_dim=output_dim, name='embed')
        stack.reset()
        outputs, states = stack.unroll(seq_len, inputs=embed,  merge_outputs=True)
        print("args:", outputs.list_arguments())
        pred = mx.sym.Reshape(outputs, shape=(-1, args.num_hidden))

        pred = mx.sym.FullyConnected(data=pred, num_hidden=args.num_hidden, name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        #label = mx.sym.Variable("label")
        print(label.debug_str)
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
        return pred, ('data',), ('softmax_label',)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    # print("___default bucketkey",data_train.default_bucket_key)
    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_train.default_bucket_key,
        context             = contexts)

    print("___eval_metric",mx.metric.Perplexity(invalid_label))

    model_prefix = "rnndemo"
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
    model_prefix = None
    checkpoint = None if model_prefix is None else mx.callback.do_checkpoint(model_prefix)

    model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = mx.metric.Perplexity(1),
        #eval_metric         = ['accuracy'],
        kvstore             = args.kv_store,
        optimizer           = args.optimizer,
        optimizer_params    = { 'learning_rate': args.lr,
                                'momentum': args.mom,
                                'wd': args.wd },
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch           = args.num_epochs,
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.disp_batches),
        epoch_end_callback = checkpoint)

    out = model.get_outputs()
    print(out[0].asnumpy())
    print(out[0].asnumpy().shape)