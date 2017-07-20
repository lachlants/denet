import os
import sys
import time
import numpy
import argparse
import math
import random
import socket
import pickle
import time
import multiprocessing as mp
import threading

import denet.common as common
import denet.common.logging as logging
import denet.dataset as dataset
import denet.multi.shared as shared

from denet.multi.worker import WorkerProcess
from denet.multi.update_server import UpdateServer
from denet.multi.update_client import UpdateClient
from denet.dataset import DatasetExportThread

#updates model with training data
def run_train_epoch(args, update_client, workers, model, train_data, learn_rate):

    import model_cnn

    logging.info("Perform train...")
    batch_size_factor = args.batch_size_factor
    output_prefix = args.output_prefix
    model_dims = args.model_dims
    model_save_dt = args.model_save_dt*60
    
    #update learning rates:
    for worker in workers:
        with worker.learn_rate.get_lock():
            worker.learn_rate.value = learn_rate

    #randomly shuffle data before each epoch, set seed to ensure each node has same data order
    random.seed(args.seed + update_client.epoch)
    train_data.shuffle()

    #perform initial sync so that all nodes have the same model
    model_update = shared.ModelUpdate(model_dims)
    model_update.import_updates(model)
    # update_client.sync(model_update, workers, initial=True)

    #get subset next
    subset_next = update_client.get_subset_next()

    #start export of data
    batch_size = len(workers) * model.batch_size * batch_size_factor
    logging.info("SGD batch size is %ix%ix%i = %i"%(batch_size_factor, len(workers), model.batch_size, batch_size))
    export_thread = DatasetExportThread(model, train_data, subset_next, batch_size, True)

    #start processing
    total_cost=0
    total_it=0
    subset_current = subset_next
    epoch_current = update_client.epoch
    for worker in workers:
        worker.set_epoch(epoch_current)

    timer = common.Timer()
    timer_save = common.Timer()
    while subset_next >= 0:

        #wait until export is ready
        timer.reset()
        export_thread.wait()
        data_x, data_y, data_size = export_thread.get_export()
        subset_current = subset_next
        del export_thread
        if timer.current() > 1:
            logging.warning("Warning: needed an additional %.1f seconds for dataset export"%timer.current())

        #print training classes for checking random seed etc
        logging.debug("Sample Metas: ", data_y[0:min(3, len(data_y))])

        #start exporting next subset
        subset_next = update_client.get_subset_next()
        if subset_next >= 0:
            export_thread = DatasetExportThread(model, train_data, subset_next, batch_size, True)

        # #store initial model before changes
        # model_update_delta = model_update.copy()

        logging.info("Evaluating training function")
        timer.reset()
        batch_num = data_x.shape[0] // model.batch_size
        it_num = batch_num // (len(workers)*batch_size_factor)
        index=0
        subset_cost = 0

        while(index < batch_num):

            total_ts = time.time()

            def train_worker_thread(worker, indexs):
                worker.wait()
                worker.model_write(model_update)
                worker.train_begin()
                for i in indexs:
                    dx = data_x[i*model.batch_size : (i + 1)*model.batch_size]
                    dy = data_y[i*model.batch_size : (i + 1)*model.batch_size]
                    worker.train_step(dx, dy)
                    worker.wait()
                worker.train_end()
                worker.model_read()
                worker.wait()
                
            threads=[]
            for worker in workers:
                worker_indexs=[]
                for _ in range(batch_size_factor):
                    if index < batch_num:
                        worker_indexs.append(index)
                        index += 1

                t = threading.Thread(target=train_worker_thread, args=(worker, worker_indexs))
                t.start()
                threads.append((t, time.time()))

            proc_ts = []
            for t, start_ts in threads:
                t.join()
                proc_ts.append(int(1000*(time.time() - start_ts)))

            #average models between GPUS and print batch info
            combine_ts = time.time()
            batch_cost = 0
            model_update.set_mean_init()
            for worker in workers:
                model_update.set_mean_update(worker.model_update)
                with worker.cost.get_lock():
                    batch_cost += worker.cost.value
            model_update.set_mean_finish()
            batch_cost /= len(workers)
            subset_cost += batch_cost
            it_index = index // (len(workers)*batch_size_factor)
            combine_ts = int(1000*(time.time() - combine_ts))

            logging.verbose("Processing times (ms):", proc_ts, ", Combine time: %i ms"%combine_ts)
            logging.info("Subset %i/%i, Batch It %i/%i"%(subset_current+1, train_data.subset_num, it_index, it_num), "- Cost:", batch_cost, "Time: %i ms"%(1000*(time.time() - total_ts)))

        logging.info("Training subset %i took %0.1f sec, mean cost:"%(subset_current+1, timer.current()), subset_cost / it_num)
        total_it += it_num
        total_cost += subset_cost

        #update with server (if one exists)
        model_update.export_updates(model)
        # model_update_delta.set_delta(model_update)
        # update_client.update(model_update_delta, model_update, workers)

        #save intermediate models
        if timer_save.current() > model_save_dt and model_save_dt > 0:
            model_cnn.save_to_file(model, output_prefix + "_epoch%03i_subset%03i.mdl.gz"%(epoch_current, subset_current+1))
            timer_save.reset()


    #perform final sync so that all nodes have the same model
    update_client.sync(model_update, workers)

    #save final models
    model_cnn.save_to_file(model, output_prefix + "_epoch%03i_final.mdl.gz"%(epoch_current))

    return (total_cost / total_it)

#compute per class error rates
def compute_error(workers, model, test_data):

    logging.info("Perform inference...")
    class_errors=[0]*model.class_num
    class_samples=[0]*model.class_num

    #export first data
    export_thread = DatasetExportThread(model, test_data, 0, len(workers)*model.batch_size, False)
    for subset in range(test_data.subset_num):

        export_thread.wait()
        data_x, data_y, data_size = export_thread.get_export()
        truth_labels = export_thread.get_labels()
        del export_thread

        #start exporting next subset
        if (subset+1) < test_data.subset_num:
            logging.info("Starting next subset export")
            export_thread = DatasetExportThread(model, test_data, subset + 1, len(workers)*model.batch_size, False)

        logging.info("Evaluating predict function (%i samples)"%data_size)
        predict_y = numpy.zeros(shape=(data_x.shape[0], model.get_probability_shape()[1]), dtype=numpy.float32)
        nbatch = data_x.shape[0] // model.batch_size
        index=0
        while(index < nbatch):

            #use group of workers to process multi gpu batch
            active_workers=[]
            for worker in workers:
                if index < nbatch:
                    dx = data_x[index * model.batch_size : (index + 1) * model.batch_size];
                    worker.predict(dx)
                    active_workers.append(worker)
                    index += 1

            #as workers finish combine resulting models
            done=[False]*len(active_workers)
            while(False in done):
                for i, worker in enumerate(active_workers):
                    if done[i] == False and worker.get_active() == 0:
                        done[i] = True
                        worker_index = index - len(active_workers) + i
                        logging.verbose("Adding predictions %i/%i"%(worker_index, nbatch))
                        with worker.data_y.lock:
                            predict_y[worker_index * model.batch_size : (worker_index + 1) * model.batch_size, ...] = worker.data_y.get_array()[...]

                time.sleep(0.0001)

        predict_y = predict_y[0:data_size, ...]
        predict_labels = numpy.argmax(predict_y, axis=1)
        #print("sizes:", truth_labels.shape, data_size)
        for i in range(data_size):
            class_samples[truth_labels[i]] += 1
            if predict_labels[i] != truth_labels[i]:
                class_errors[truth_labels[i]] += 1

    #compute errors
    error=100.0*sum(class_errors) / sum(class_samples)
    cls_errors=[]
    for i in range(model.class_num):
        x = 100.0*class_errors[i] / class_samples[i] if class_samples[i] > 0 else 100.0
        cls_errors.append((i, x, class_samples[i]))

    return (error, cls_errors)

def save_results(fname, error, class_errors):
    with open(fname, "w") as f:
        logging.info("Overall Error=%.2f%%"%(error), file=f)
        for d in class_errors:
            logging.info("Class %i=%.2f%% (%i samples)"%(d[0],d[1],d[2]*d[1]/100), file=f)

def load_restart_args(args_fname, args):

    if not os.path.isfile(args_fname):
        raise Exception("Cannot find arguments file:" + args_fname)

    logging.info("Loading arguments from:", args_fname)
    with open(args_fname, "rb") as f:
        args = pickle.load(f)

    #search for models
    model_fnames = common.find_files(os.path.dirname(args.output_prefix),  "*_epoch*.mdl.gz")
    if len(model_fnames) == 0:
        raise Exception("Could not find any intermediate models to continue training from!")

    v = os.path.basename(model_fnames[-1])
    v = v[:v.find(".")].split("_")

    if v[-1] == "final":
        args.epoch_start = int(v[-2][5:]) + 1
        args.subset_start = 0
    else:
        args.epoch_start = int(v[-2][5:])
        args.subset_start = int(v[-1][6:]) + 1

    args.model = model_fnames[-1]
    logging.info("Continuing training with model:", args.model, "epoch:", args.epoch_start, "subset:", args.subset_start)
    return args


def main():

    #load arguments:
    parser = argparse.ArgumentParser(description='Train a convolutional network using labelled data')
    logging.add_arguments(parser)
    parser.add_argument("--use-acc-mode", default=False, action="store_true", help="Use model accumulation over multiple batches (uses more VRAM)")
    parser.add_argument("--cost-factors", default=[], nargs="+", help="Multiplicative factors for model costs")
    parser.add_argument("--export-model-dims", default=False, action="store_true", help="Ignore, don't use this option!")
    parser.add_argument("--model-dims", default="./model-dims.json", type=str, help="export file for shared model dimensions")
    parser.add_argument("--model-save-dt", default=30, type=int, help="Minimum time (min) between saving an intermediate model. Use 0 to disable.")
    parser.add_argument("--model", required=False, default=None, help="Model to continue training.")
    parser.add_argument("--gpus", nargs="+", default=["gpu0"], help="list of gpus to train over")
    parser.add_argument("--update-server", metavar="<addr> [port] [offset] [delta]", nargs="+", default=None, 
                        help="model update server for synchronizing multiple networked machines. Set <addr> to 'mpi' for MPI networking.")
    parser.add_argument("--subset-max", type=int, default=10000000, help="Specify maximum number of subsets to be used in each training epoch")
    parser.add_argument("--train", default=None, help="The folder with training / validation data")
    parser.add_argument("--test", default=None, help="The folder with testing data (optional)")
    parser.add_argument("--test-mode", default="default", help="Testing Mode")
    parser.add_argument("--test-epochs", type=int, default=1, help="Epochs between each test evaluation")
    parser.add_argument("--thread-num", type=int, default=1, help="Number of threads to use for supported opeartions (e.g. loading/distorting datasets)")
    parser.add_argument("--extension", default="ppm", help="Image file extension")
    parser.add_argument("--activation", default="relu", help="Activation function used in convolution / hidden layers (tanh, relu, leaky-relu)")
    parser.add_argument("--border-mode", default="half", help="Border mode for convolutional layers (full, valid)")
    parser.add_argument("--output-prefix", default="./model", help="Output prefix for model files")
    parser.add_argument("--solver", type=str, default="nesterov", help="")
    parser.add_argument("--weight-init", nargs="+", default=["he-backward"], help="Weight initialization scheme")
    parser.add_argument("--initial-tune", type=float, default=0.0, help="Perform initial tuning with learning rate")
    parser.add_argument("--learn-rate", type=float, default=0.1, help="Learning rate for weights and biases.")
    parser.add_argument("--learn-momentum", type=float, default=[0.0,0.0], nargs="+", help="Learning momentum for weights and biases (0.0 - 1.0).")
    parser.add_argument("--learn-anneal", type=float, default=1, help="Annealing factor per epoch for weight and bias learning rate")
    parser.add_argument("--learn-anneal-epochs", nargs="+", type=int, default=[], help="Epochs to apply learning rate annealing (default every epoch)")
    parser.add_argument("--learn-decay", type=float, default=0.0, help="L2 weight decay (not applied to biases). ")
    parser.add_argument("--epochs", type=int, default=30, help="The number of training epochs")
    parser.add_argument("--epoch-start", type=int, default=0, help="Epoch to start from")
    parser.add_argument("--subset-start", type=int, default=0, help="Subset to start from")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to load from training set")
    parser.add_argument("--batch-size", type=int, default=32, help="Size of each processing batch (per GPU)")
    parser.add_argument("--batch-size-factor", type=int, default=1, help="Batch size multiplier, use when desired batch size won't fit in memory.")
    parser.add_argument("--batch-data-size", type=int, default=1, help="Number of batches to upload to GPU for processing")
    parser.add_argument("--seed", type=int, default=23455, help="Random Seed for weights")
    parser.add_argument("--split-seed", type=int, default=0, help="Random Seed for splitting into validation / training")
    parser.add_argument("--export-symbolic", default=None, help="Save datasets as symbolic links")
    parser.add_argument("--distort-mode", default=[], nargs="+", help="Distortions to apply to training data (default, cifar10, disable)")
    parser.add_argument("--augment-mirror", default=False, action="store_true", help="Augment training data with horizontally mirrored copies")
    parser.add_argument("--skip-train", default=False, action="store_true", help="Skip training of model")
    parser.add_argument("--skip-layer-updates", type=int, nargs="+", default=[], help="Skip training updates to specified layers")
    parser.add_argument("--model-desc", default=["C100,7", "P2", "C150,4", "P2", "C250,4", "P2", "C300,1", "CR"], nargs="+", type=str, help="Network layer description" )
    parser.add_argument("--theano-flags", type=str, default="lib.cnmem=1.0", help="Additional THEANO_FLAGS environment variables for worker threads")
    parser.add_argument("--restart", default=False, action="store_true", help="Restart training of model")
    args = parser.parse_args()

    logging.init(args)

    #continue training
    args_fname = "./train.args"
    if args.restart:
        args = load_restart_args(args_fname, args)
    else:
        logging.info("Exporting arguments:", args_fname)
        with open(args_fname, "wb") as f:
            pickle.dump(args, f)

    #start MPI update server if this is master node:
    if not args.update_server is None and args.update_server[0] == "mpi":
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() == 0:
            momentum = float(args.update_server[1])
            update_server = UpdateServer(args.model_dims, momentum=momentum, use_mpi=True, use_async=True)
            sys.exit(update_server.start())

    #set random seeds
    random.seed(args.seed)
    numpy.random.seed(args.seed)

    #load training dataset
    logging.info("Loading training data: " + str(args.train))
    train_data = dataset.load(args.train, args.extension, is_training=True, thread_num=args.thread_num)
    data_shape = train_data.get_data_shape()
    class_num = train_data.get_class_num()
    class_labels = train_data.class_labels
    logging.info("Found %i samples across %i class Labels:\n"%(train_data.get_total_size(), class_num), class_labels)

    #HACK to determine model parameter dimensions for shared models without initializing theano...
    #Not need any more in theano-0.8.0
    if not os.path.isfile(args.model_dims):
        logging.info("Exporting model dims file to " + args.model_dims)
        import model_cnn
        model = model_cnn.initialize(args, data_shape, class_labels, class_num)
        model.build_train_func(args.solver, skip_build=True)
        shared.ModelUpdate.save_dims(args.model_dims, model)
        logging.info("Done")
        exit(0)

    #construct worker processes (must be done before model due to Theano init! No longer true in theano 0.8.0):
    logging.info("Initializing worker procs for", args.gpus)
    workers = [WorkerProcess(gpu, args, data_shape, class_labels) for gpu in args.gpus]

    #initialize model (and Theano)
    import model_cnn
    model = model_cnn.initialize(args, data_shape, class_labels, class_num)
    model.build_train_func(args.solver, skip_build=True)

    #mirror training data
    if args.augment_mirror:
        train_data.augment_mirror();

    #load test dataset
    if args.test:
        logging.info("Loading test: " + str(args.test))
        test_data = dataset.load(args.test, args.extension, is_training=False, class_labels=class_labels, thread_num=args.thread_num)
        logging.info("Testing: " + str(test_data.get_total_size()) + " samples")
        assert(test_data.get_total_size() != 0)

    #connect with update server
    if not args.update_server is None:

        addr = args.update_server[0]
        use_mpi = bool(addr == "mpi")
        use_async = bool(len(args.update_server) == 2)
        port = 0 if use_mpi else int(args.update_server[1])
        offset = 0 if use_async else int(args.update_server[2])
        delta = 0 if use_async else int(args.update_server[3])

        logging.info("Connecting to update server (async=%i, mpi=%i): "%(use_async, use_mpi), addr, port)
        sock = 0 if use_mpi else socket.create_connection((addr, port))
        update_client = UpdateClient(args.epoch_start, args.subset_start, train_data.subset_num, sock, use_async, use_mpi, offset, delta)
    else:
        update_client = UpdateClient(args.epoch_start, args.subset_start, train_data.subset_num)

    #perform training and save models
    if args.initial_tune > 0:
        logging.info("----- Initial Fine Tune -----")
        logging.info("Running initial tune with learning rate:", args.initial_tune)
        run_train_epoch(args, update_client, workers, model, train_data, args.initial_tune)

    #anneal learning rate
    learn_rate = args.learn_rate
    for epoch in range(0, args.epoch_start):
        if len(args.learn_anneal_epochs) == 0 or (epoch+1) in args.learn_anneal_epochs:
            logging.verbose("Annealing learning rate")
            learn_rate *= args.learn_anneal

    #Run training
    best_test_error=100.0;
    for epoch in range(args.epoch_start, args.epochs):
        logging.info("----- Training Epoch: " + str(epoch) + " -----")

        #perform training and save models
        if not args.skip_train:
            logging.info("Training with learning rates " + str(learn_rate) + " and momentum " + str(args.learn_momentum))
            timer = common.Timer()
            cost = run_train_epoch(args, update_client, workers, model, train_data, learn_rate)
            logging.info("Training - mean cost:", cost, ", took %.0f sec"%timer.current())

        #anneal learning rate
        if len(args.learn_anneal_epochs) == 0 or (epoch+1) in args.learn_anneal_epochs:
            logging.verbose("Annealing learning rate")
            learn_rate *= args.learn_anneal

        #perform testing
        test_error=0
        if args.test and ((epoch%args.test_epochs) == 0 or epoch == (args.epochs-1)):
            ts = time.time()
            test_error, test_class_errors = compute_error(workers, model, test_data)
            logging.info("Epoch %i Test Error: %.2f%%, Took %.0f sec"%(epoch, test_error, time.time()-ts))
            save_results(args.output_prefix + "_epoch%03i.test"%epoch, test_error, test_class_errors)

    logging.info("Finished Training")

if __name__ == '__main__':
    #make additional processes spawn a new python interpretter
    import multiprocessing as mp
    mp.set_start_method('spawn')
    sys.setrecursionlimit(10000)
    sys.exit(main())
