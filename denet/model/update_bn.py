import os
import sys
import gzip
import time
import numpy
import pickle
import argparse
import math
import random
import theano

import denet.common.logging as logging
import denet.dataset as dataset
import denet.model.model_cnn as model_cnn
import denet.layer

def main(argv=None):

    #load arguments:
    parser = argparse.ArgumentParser(description='Predict labels using trained model')
    logging.add_arguments(parser)
    parser.add_argument("--model", required=True, help="input model file")
    parser.add_argument("--output", required=True, help="output model file")
    parser.add_argument("--input", required=True, help="The folder with data")
    parser.add_argument("--extension", default="png", help="Image file extension")
    parser.add_argument("--batch-size", type=int, default=128, help="Size of processing batchs")
    parser.add_argument("--thread-num", default=4, type=int, help="Number of threads for dataset loading")
    args = parser.parse_args()
    
    logging.init(args)

    logging.info("Loading model: ", args.model)
    model = model_cnn.load_from_file(args.model, args.batch_size)
    logging.info("Class labels:\n", model.class_labels)

    logging.info("Loading Dataset...")
    data = dataset.load(args.input, args.extension, is_training=True, thread_num=args.thread_num)
    data.shuffle()
    data.load_from_subset(0)
    dataset_p = data.prepare(model.width, model.height, model.image_border, model.distort_mode, True, copy_data=False)
    dataset_x, _, dataset_size = dataset_p.export(args.batch_size)

    #find all batch norm layers
    bn_layers = []
    for layer in model.layers:
        if layer.type_name in ["batchnorm", "batchnorm-relu"]:
            bn_layers.append(layer)
        elif layer.type_name == "resnet":
            for l in layer.layers:
                if l.type_name in ["batchnorm", "batchnorm-relu"]:
                    bn_layers.append(l)

    logging.info("Found %i batch norm layers"%len(bn_layers))

    for i,layer in enumerate(bn_layers):
        bn_func = theano.function([model.input], [layer.input.mean(axis=[0,2,3]), layer.input.var(axis=[0,2,3])], givens=[(denet.layer.get_train(), theano.tensor.cast(0, 'int8'))])
        bn_mean = numpy.zeros((layer.input_shape[1], ), numpy.float64)
        bn_var = numpy.zeros((layer.input_shape[1], ), numpy.float64)

        logging.info("Estimating mean and var for layer %i with %i samples"%(i, dataset_size))
        n = dataset_size // args.batch_size
        for b in range(n):
            u,v = bn_func(dataset_x[b*args.batch_size:(b+1)*args.batch_size])
            bn_mean += u
            bn_var += v

        eps = 1e-5
        bn_mean = (bn_mean / n).astype(numpy.float32)
        bn_var = (bn_var / n).astype(numpy.float32)
        bn_invstd = 1.0 / (numpy.sqrt(bn_var + eps))

        logging.verbose("Layer %i - Old Mean:"%i, layer.mean.get_value())
        logging.verbose("Layer %i - New Mean:"%i, bn_mean)
        logging.verbose("Layer %i - Old Std:"%i, layer.stdinv.get_value())
        logging.verbose("Layer %i - New Std:"%i, bn_invstd)

        layer.mean.set_value(bn_mean, borrow=True)
        layer.stdinv.set_value(bn_invstd, borrow=True)

    model_cnn.save_to_file(model, args.output)
    logging.info("Done")

if __name__ == '__main__':
    sys.exit(main())
