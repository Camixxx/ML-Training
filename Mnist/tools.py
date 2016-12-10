import  mxnet as mx
import argparse

# Load the iterator of the mnist
def get_iterator(data_shape):
    def get_iterator_impl(args, kv):
        data_dir = args.data_dir
        if '://' not in args.data_dir:
           #_download(args.data_dir)
            print("Missint the mnist dataset,please download it")
        flat = False if len(data_shape) == 3 else True

        train           = mx.io.MNISTIter(
            image       = data_dir + "train-images-idx3-ubyte",
            label       = data_dir + "train-labels-idx1-ubyte",
            input_shape = data_shape,
            batch_size  = args.batch_size,
            shuffle     = True,
            flat        = flat,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)

        val = mx.io.MNISTIter(
            image       = data_dir + "t10k-images-idx3-ubyte",
            label       = data_dir + "t10k-labels-idx1-ubyte",
            input_shape = data_shape,
            batch_size  = args.batch_size,
            flat        = flat,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)

        return (train, val)
    return get_iterator_impl


def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='mlp',
                        choices = ['mlp', 'lenet', 'lenet-stn'],
                        help = 'the cnn to use')
    parser.add_argument('--data-dir', type=str, default='demo/mnist/',
                        help='the input data directory')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
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
