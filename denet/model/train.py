import os
import sys
import gzip
import time
import numpy
import argparse
import math
import random
import copy
import theano

import denet.model.model_cnn as model_cnn
import denet.dataset as dataset
import denet.common.logging as logging
import denet.common as common

#compute per class error rates
def compute_error(data, model):

    class_errors=[0]*model.class_num
    class_samples=[0]*model.class_num
    for subset in range(data.subset_num):
        data.load_from_subset(subset)

        #logging.info("Preparing test data")
        #test_data = data.prepare(model.width, model.height, model.image_border, model.distort_mode, training=False)
 
        logging.info("Computing error...")
        labels_predict = model.predict_label(data)
        labels = data.get_labels()
        for i in range(len(data)):
            class_samples[labels[i]] += 1
            if labels_predict[i] != labels[i]:
                class_errors[labels[i]] += 1

    error = 100.0*sum(class_errors) / sum(class_samples)
    class_errors = [(i, 100.0*class_errors[i] / class_samples[i], class_samples[i]) for i in range(model.class_num)]
    return (error, class_errors)

def save_results(fname, error, class_errors):
    with open(fname, "w") as f:
        print("Overall Error=%.2f%%"%(error), file=f)
        for d in class_errors:
            print("Class %i=%.2f%% (%i samples)"%(d[0],d[1],d[2]*d[1]/100), file=f)

def main():

    #load arguments:
    parser = argparse.ArgumentParser(description='Train a convolutional network using labelled data.')
    logging.add_arguments(parser)
    parser.add_argument("--model", required=False, default=None, help="Model to continue training.")
    parser.add_argument("--cost-factors", default=[], nargs="+", help="Multiplicative factors for model costs")
    parser.add_argument("--thread-num", type=int, default=1, help="Number of threads to use for supported opeartions (e.g. loading/distorting datasets)")
    parser.add_argument("--extension", default="ppm", help="Image file extension")
    parser.add_argument("--train", default=None, help="The folder with training / validation data")
    parser.add_argument("--test", default=None, help="The folder with testing data (optional)")
    parser.add_argument("--test-epochs", type=int, default=1, help="Epochs between each test evaluation")
    parser.add_argument("--test-mode", default="default", help="Mode to use for testing")
    parser.add_argument("--border-mode", default="valid", help="Border mode for convolutional layers (full, valid)")
    parser.add_argument("--output-prefix", default="./model", help="Output prefix for model files")
    parser.add_argument("--activation", default="relu", help="Activation function used in convolution / hidden layers (tanh, relu, leaky-relu)")
    parser.add_argument("--solver", type=str, default="nesterov", help="")
    parser.add_argument("--weight-init", nargs="+", default=["he-backward"], help="Weight initialization scheme")
    parser.add_argument("--learn-rate", type=float, default=0.1, help="Learning rate for weights and biases.")
    parser.add_argument("--learn-momentum", type=float, default=[0.0, 0.0], nargs="+", help="Learning momentum for weights and biases (0.0 - 1.0).")
    parser.add_argument("--learn-anneal", type=float, default=1, help="Annealing factor per epoch for weight and bias learning rate")
    parser.add_argument("--learn-anneal-epochs", nargs="+", type=int, default=[], help="Epochs to apply learning rate annealing (default every epoch)")
    parser.add_argument("--learn-decay", type=float, default=0.0, help="L2 weight decay (not applied to biases). ")
    parser.add_argument("--epochs", type=int, default=30, help="The number of training epochs")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to load from training set")
    parser.add_argument("--batch-size", type=int, default=32, help="Size of processing batchs")
    parser.add_argument("--seed", type=int, default=23455, help="Random Seed for weights")
    parser.add_argument("--distort-mode", default=[], nargs="+", help="Distortions to apply to training data (default, cifar10, disable)")
    parser.add_argument("--disable-intermediate", default=False, action="store_true", help="Disable outputting of intermediate model files")
    parser.add_argument("--augment-mirror", default=False, action="store_true", help="Augment training data with horizontally mirrored copies")
    parser.add_argument("--skip-train", default=False, action="store_true", help="Skip training of model")
    parser.add_argument("--skip-layer-updates", type=int, nargs="+", default=[], help="Skip training updates to specified layers")
    parser.add_argument("--model-desc", default=["C[100,7]", "P[2]", "C[150,4]", "P[2]", "C[250,4]", "P[2]", "C[300,1]", "R"], nargs="+", type=str, help="Network layer description" )
    args = parser.parse_args()

    logging.init(args)

    #set random seeds
    random.seed(args.seed)
    numpy.random.seed(args.seed)

    #load training dataset
    logging.info("Loading training data:", args.train)
    train_data = dataset.load(args.train, args.extension, is_training=True, thread_num=args.thread_num)
    data_shape = train_data.get_data_shape()
    class_num = train_data.get_class_num()
    class_labels = train_data.class_labels
    logging.info("Found %i class labels:\n"%class_num, class_labels)

    #hack for reducing training data size
    if not args.max_samples is None:
        train_data.data = random.sample(train_data.data, args.max_samples)

    #mirror training data
    if args.augment_mirror:
        train_data.augment_mirror();

    logging.info("Training: %i samples"%len(train_data))

    #load test dataset
    if args.test:
        logging.info("Loading test: " + args.test)
        test_data = dataset.load(args.test, args.extension, is_training=False, thread_num=args.thread_num, class_labels=class_labels)

    #initialize model
    model = model_cnn.initialize(args, data_shape, class_labels, class_num)
    model.build_train_func(args.solver, args.cost_factors)

    #Run training
    best_test_error=100.0
    learn_rate = args.learn_rate
    for epoch in range(args.epochs):
        logging.info("----- Training Epoch: %i -----"%epoch)

        #perform training
        if not args.skip_train:

            logging.info("Training with solver " + args.solver +  ", learning rate " + str(learn_rate) + " and momentum " + str(args.learn_momentum))

            #shuffle dataset:
            train_data.shuffle()

            for subset in range(train_data.subset_num):
                timer = common.Timer()
                train_data.load_from_subset(subset)

                logging.info("Performing Gradient Descent...")
                cost = model.train_epoch(train_data, epoch, learn_rate, args.learn_momentum, args.learn_decay)

                nbatch = math.ceil(len(train_data) / model.batch_size)
                logging.info("Training subset %i - Cost: %.3f, Took %.1f sec"%(subset, cost, timer.current()))

        if len(args.learn_anneal_epochs) == 0 or (epoch+1) in args.learn_anneal_epochs:
            logging.verbose("Annealing learning rate")
            learn_rate *= args.learn_anneal

        #perform testing
        test_error=0
        if not args.test is None and ((epoch%args.test_epochs) == 0 or epoch == (args.epochs-1)):
            test_error, test_class_errors = compute_error(test_data, model)
            logging.info("Epoch %i test error: %.2f%% (%i samples)"%(epoch, test_error, int(test_error*len(test_data)/100.0)))
            save_results(args.output_prefix + "_epoch%03i.test"%epoch, test_error, test_class_errors)
        
        #save intermediate models
        if not args.disable_intermediate:
            model_cnn.save_to_file(model, args.output_prefix + "_epoch%03i.mdl.gz"%(epoch))


    #save final model
    model_cnn.save_to_file(model, args.output_prefix + "_epoch%03i_final.mdl.gz"%epoch)
    logging.info("Finished Training")

if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    sys.exit(main())
