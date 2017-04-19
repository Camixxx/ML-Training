import mxnet as mx
import os, sys
import logging
import tools
import argparse
import read_xml
def get_mlp():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp


# Fit

def fit(args, network , batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    # if 'log_file' in args and args.log_file is not None:
    #     log_file = args.log_file
    #     log_dir = args.log_dir
    #     log_file_full_name = os.path.join(log_dir, log_file)
    #     if not os.path.exists(log_dir):
    #         os.mkdir(log_dir)
    #     logger = logging.getLogger()
    #     handler = logging.FileHandler(log_file_full_name)
    #     formatter = logging.Formatter(head)
    #     handler.setFormatter(formatter)
    #     logger.addHandler(handler)
    #     logger.setLevel(logging.DEBUG)
    #     logger.info('start with arguments %s', args)
    # else:
    #     logging.basicConfig(level=logging.DEBUG, format=head)
    #     logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}
        # TODO: check epoch_size for 'dist_sync'
        epoch_size = args.num_examples / args.batch_size
        model_args['begin_num_update'] = epoch_size * args.load_epoch

    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    (train, val) = load_data(args)

    # train
    devs = mx.cpu()

    epoch_size = args.num_examples / args.batch_size

    # if 'lr_factor' in args and args.lr_factor < 1:
    #     model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
    #         step = max(int(epoch_size * args.lr_factor_epoch), 1),
    #         factor = args.lr_factor)
    #
    # if 'clip_gradient' in args and args.clip_gradient is not None:
    #     model_args['clip_gradient'] = args.clip_gradient
    #
    # # disable kvstore for single device
    # if 'local' in kv.type and (
    #         args.gpus is None or len(args.gpus.split(',')) is 1):
    #     kv = None

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                  = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args)

    eval_metrics = ['accuracy']
    ## TopKAccuracy only allows top_k > 1
    for top_k in [5, 10, 20]:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k = top_k))

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))

    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = checkpoint)


if __name__ == '__main__':
    args = tools.parse_args()

    # data shape and model train
    data_shape = (784, )
    net = get_mlp()
    #data_loader = tools.get_iterator(data_shape)

    # train
    fit(args, net)



def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='mlp',
                        choices = ['mlp', 'lenet', 'lenet-stn'],
                        help = 'the cnn to use')
    parser.add_argument('--data-dir', type=str, default='demo/mnist/',
                        help='the input data directory')
    # parser.add_argument('--gpus', type=str,
    #                     help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.1,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,default='demo/',
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    return parser.parse_args()