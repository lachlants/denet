import time
import sys
import os
import json
import numpy
import ctypes
import threading
import psutil
import queue
import math
import multiprocessing as mp

import denet.common.logging as logging
import denet.multi.shared as shared

#flush logs every 5 seconds
def flush_logs():
    sys.stdout.flush()
    sys.stderr.flush()
    threading.Timer(5.0, flush_logs).start()

#runs in seperate process - must be in __main__ for multiprocessing "spawn" mode to work
def run_worker(gpu, args, data_shape, class_labels, class_num, task_queue, active, epoch, learn_rate, cost, timer,
    data_x, data_y, data_m, model_update):

    #redirect output (unbuffered)
    sys.stdout = open(gpu + ".out", 'w')
    sys.stderr = open(gpu + ".err", 'w')
    logging.init(args, flush=True)
    sys.setrecursionlimit(10000)
    
    #create thread to flush stdout / stderr every 5 seconds
    flush_logs()

    logging.info(gpu + ": initializing")

    #remove all openmpi variables!
    for v in os.environ.keys():
        if v[:5] == "OMPI_":
            del os.environ[v]

    #set compile dir and gpu (possible since theano hasn't been imported yet!)
    if not "THEANO_FLAGS" in os.environ:
        os.environ["THEANO_FLAGS"] = ""

    import socket
    os.environ["THEANO_FLAGS"] += "," + args.theano_flags + ","
    os.environ["THEANO_FLAGS"] += "device=" + gpu + ",";
    os.environ["THEANO_FLAGS"] += "force_device=True,";
    os.environ["THEANO_FLAGS"] += "compiledir=~/.theano/" + socket.gethostname() + "-" + gpu + "/,"
    #os.environ["THEANO_FLAGS"] += "lib.cnmem=1,";
    os.environ["THEANO_FLAGS"] += "nvcc.flags=-D_FORCE_INLINES,"
    logging.info(gpu + ": Using THEANO_FLAGS:", os.environ["THEANO_FLAGS"])

    #initialize local model
    import denet.model.model_cnn as model_cnn
    model = model_cnn.initialize(args, data_shape, class_labels, class_num)

    #pre-initialize training function
    use_acc_mode = args.batch_size_factor > 1 and args.use_acc_mode
    model.build_train_func(args.solver, args.cost_factors, use_acc_mode=use_acc_mode)
    if use_acc_mode:
        train_begin_func = model.func["train_begin"]
        train_end_func = model.func["train_end"]

    #begin processing loop
    iteration = 0
    while(True):

        #try to start next task immediately otherwise wait for task
        wait_time = time.time()
        try:
            task = task_queue.get(block=False)
        except queue.Empty:
            logging.verbose(gpu + ": waiting for task")
            with active.get_lock():
                active.value = 0
            task = task_queue.get(block=True)
        wait_time = time.time() - wait_time

        logging.verbose(gpu + ": " + task + " (wait time=%i ms)"%(1000*wait_time))

        #calculate updates
        ts = time.time()
        if task == "predict":
            with data_x.lock, data_y.lock:
                data_y.get_array()[...] = model.predict_output_step(data_x.get_array())
        
        elif task == "model-read":
            model_update.import_updates(model)
          
        elif task == "model-write":
            model_update.export_updates(model)

        elif task == "train-begin":          
            if use_acc_mode:
                train_begin_func()

            with cost.get_lock():
                cost.value = 0

        elif task == "train-step":
            with cost.get_lock(), epoch.get_lock(), learn_rate.get_lock(), data_x.lock:
                data_meta = data_m.get(block=True)
                c, _ = model.train_step(data_x.get_array(), data_meta, epoch.value, iteration, 
                                        learn_rate.value, args.learn_momentum, args.learn_decay)
                if math.isnan(c):
                    raise Exception("Encountered NaN cost for worker")

                cost.value += c

            iteration += 1

        elif task == "train-end":
            if use_acc_mode:
                train_end_func()

            with cost.get_lock():
                cost.value /= args.batch_size_factor

        elif task == "done":
            exit(0)
          
        with timer.get_lock():
            timer.value = time.time() - ts
            logging.info(gpu + ": %s took %i ms"%(task, 1000*timer.value))

def run_worker_wrapper(gpu, args, data_shape, class_labels, class_num, task_queue, active, epoch, learn_rate, cost, timer,
                       data_x, data_y, data_m, model_update, error_q):
    try:
        run_worker(gpu, args, data_shape, class_labels, class_num, task_queue, active, epoch, learn_rate, cost, timer,
                   data_x, data_y, data_m, model_update)

    except Exception as e:
        error_q.put(e)
        raise(e)

class WorkerProcess:
    def __init__(self, gpu, args, data_shape, class_labels):
        super().__init__()

        self.gpu = gpu
        self.args = args
        self.class_labels = class_labels
        self.data_shape = data_shape
        self.class_num = len(class_labels)

        #shared variables
        self.task_queue = mp.Queue()
        self.active = mp.Value('i', 1)
        self.epoch = mp.Value('i', 0)
        self.learn_rate = mp.Value('f', args.learn_rate)
        self.cost = mp.Value('f', 0.0)
        self.timer = mp.Value('f', 0.0)

        self.model_update = shared.ModelUpdate(args.model_dims, args.batch_size)
        self.data_x = shared.Array(self.model_update.input_shape)
        self.data_y = shared.Array(self.model_update.output_shape)
        self.data_m = mp.Queue()
        self.error_q = mp.Queue()

        logging.info("Starting worker:" + self.gpu)
        proc_args = (self.gpu, self.args, self.data_shape, self.class_labels, self.class_num, self.task_queue, self.active, self.epoch, self.learn_rate, self.cost,
                     self.timer, self.data_x, self.data_y, self.data_m, self.model_update, self.error_q)
        self.proc = mp.Process(target=run_worker_wrapper, args=proc_args, name=self.gpu)
        self.proc.daemon = True
        self.proc.start()
        self.ps_proc = psutil.Process(self.proc.pid)

    def set_active(self, state):
        with self.active.get_lock():
            self.active.value = state

    def set_epoch(self, index):
        with self.epoch.get_lock():
            self.epoch.value = index

    def get_active(self):
        v=0
        with self.active.get_lock():
            v=self.active.value
        return v

    def start_task(self, task):
        self.set_active(1)
        self.task_queue.put(task)

    def train_begin(self):
        self.start_task("train-begin")

    def train_step(self, data_x, data_m):
        with self.data_x.lock:
            self.data_x.get_array()[...] = data_x[...]
        self.data_m.put(data_m)
        self.start_task("train-step")

    def train_end(self):
        self.start_task("train-end")

    #read model off GPU into self.model_update
    def model_read(self):
        self.start_task("model-read")

    #write model in shared update onto GPU
    def model_write(self, shared_update):
        self.model_update.set_updates(shared_update)
        self.start_task("model-write")

    def predict(self, data_x):
        with self.data_x.lock:
            self.data_x.get_array()[...] = data_x[...]
        self.start_task("predict")

    def done(self):
        self.start_task("done")
        self.proc.join()

    #returns number of seconds waited
    def wait(self, dt = 0.001):
        count = 0
        done = False
        while(not done):
            with self.active.get_lock():
                if self.active.value == 0:
                    done=True

            #check for exceptions
            try:
                error = self.error_q.get(block=False)
                self.proc.terminate()
                raise(error)
                exit(1)
            except queue.Empty:
                pass
            
            #check if process has stopped
            # if self.ps_proc.status() == psutil.STATUS_ZOMBIE or self.ps_proc.status() == psutil.STATUS_DEAD:
            #     raise Exception("Worker died prematurely: " + self.gpu)

            time.sleep(dt)
            count += 1

        return count*dt

