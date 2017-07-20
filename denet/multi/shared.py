import json
import numpy
import ctypes
import multiprocessing as mp
import threading
import time

import denet.common as common

#class for sharing numpy arrays between processes
class Array:
    def __init__(self, shape, dtype = numpy.float32):
        num_elems = numpy.prod(shape)

        if dtype == numpy.int32:
            c_type = ctypes.c_int
        elif dtype == numpy.float32:
            c_type = ctypes.c_float
        elif dtype == numpy.float64:
            c_type = ctypes.c_double
        else:
            assert(0)

        #shared storage for numpy array
        self.shape = shape
        self.dtype = dtype
        self.base = mp.RawArray(c_type, int(num_elems))
        self.lock = mp.RLock()

    #overloaded operators for convienince
    def add_array(self, p):
        with p.lock, self.lock:
            self.get_array()[...] += p.get_array()[...]

    def set_array(self, p):
        with p.lock, self.lock:
            self.get_array()[...] = p.get_array()[...]

    def fill_value(self, v):
        with self.lock:
            self.get_array().fill(0)

    def mul_value(self, v):
        with self.lock:
            self.get_array()[...] *= v

    def div_value(self, v):
        self.mul_value(1.0 / v)

    def get_array(self):
        #when lock is true, base doesn't have get_obj
        array = numpy.frombuffer(self.base, dtype=self.dtype)
        return array.reshape(self.shape)

    #serialize numpy array
    def export_json(self):
        with self.lock:
            r = common.numpy_to_json(self.get_array())
        return r

    def import_json(self, data):
        with self.lock:
            self.get_array()[...] = common.numpy_from_json(data)

#class for sharing model parameters between processes
class ModelUpdate:
    def __init__(self, fname, batch_size=None):

        self.dims_fname = fname
        with open(fname, "r") as f:
            json_data = json.load(f)
            if batch_size is None:
                batch_size = json_data["input"][0]

            self.input_shape = tuple([batch_size] + json_data["input"][1:])
            self.output_shape = tuple([batch_size] + json_data["output"][1:])
            self.updates=[Array(dim["shape"], numpy.float32) for dim in json_data["dims"]]
                

    def copy(self):
        r = ModelUpdate(self.dims_fname)
        for i in range(len(self.updates)):
            r.updates[i].set_array(self.updates[i])
        return r

    #debugging!
    def get_samples(self):
        return [update.get_array().flatten()[0] for update in self.updates]
    #
    def set_updates(self, shared_params):
        for i,p in enumerate(self.updates):
            p.set_array(shared_params.updates[i])

    def add_delta(self, model_update, alpha = 1.0):
        for i in range(len(self.updates)):
            with self.updates[i].lock, model_update.updates[i].lock:
                self.updates[i].get_array()[...] += alpha * model_update.updates[i].get_array()[...]

    def set_delta(self, model_update):
        for i in range(len(self.updates)):
            with self.updates[i].lock, model_update.updates[i].lock:
                self.updates[i].get_array()[...] -= model_update.updates[i].get_array()[...]

    #set all values to zero
    def set_mean_init(self):
        self.update_num=0
        for i in range(len(self.updates)):
            self.updates[i].fill_value(0.0)

    #accumulate parameters
    def set_mean_update(self, shared_params):
        self.update_num += 1
        for i in range(len(self.updates)):
            self.updates[i].add_array(shared_params.updates[i])

    #divide by number of values
    def set_mean_finish(self):
        for i in range(len(self.updates)):
            self.updates[i].div_value(self.update_num)

    #calculate mean of shared_params_list and store self
    def set_mean(self, shared_params_list, nthreads=1):

        def set_mean_delta(index):
            self.updates[index].fill_value(0.0)
            for shared_params in shared_params_list:
                self.updates[index].add_array(shared_params.updates[index])
            self.updates[index].div_value(len(shared_params_list))

        if nthreads <= 1:
            for index in range(len(self.updates)):
                set_mean_delta(self, index)
        else:
            index=0
            while index < len(self.updates):

                workers=[]
                for _ in range(nthreads):
                    if index < len(self.updates):
                        workers.append(threading.Thread(target=set_mean_delta, args=(index,)))
                        workers[-1].start()
                        index += 1
                for worker in workers:
                    worker.join()

    #running average
    def set_moving_mean(self, shared_params, momentum = 0.9):
        assert len(self.updates) == len(shared_params.updates)
        for i in range(len(self.updates)):
            shared_params.updates[i].mul_value(1.0 - momentum)
            self.updates[i].mul_value(momentum)
            self.updates[i].add_array(shared_params.updates[i])

    #import parameters from model into shared array
    def import_updates(self, model):
        for i,update in enumerate(model.updates):
            with self.updates[i].lock:
                self.updates[i].get_array()[...] = update[0].get_value(borrow=False)[...]

    #export parameters from shared array into model
    def export_updates(self, model):
        index=0
        for i,update in enumerate(model.updates):
            with self.updates[i].lock:
                update[0].set_value(self.updates[i].get_array(), borrow=False)

    #save dimensions of model parameters to file
    def save_dims(fname, model):

        #make sure updates are valid
        json_dims = []
        for update in model.updates:
            json_dims.append({"shape" : update[0].get_value(borrow=True).shape})

        with open(fname, "w") as f:
            json.dump({"input" : model.get_input_shape(),
                       "output" : model.get_output_shape(),
                       "dims" : json_dims}, f)

    #serialization
    def export_npz(self, fp):
        numpy.savez(fp, *[update.get_array() for update in self.updates])

    def import_npz(self, fp):
        output = numpy.load(fp)
        assert len(output.files) == len(self.updates)
        for i,update in enumerate(self.updates):
            with update.lock:
                update.get_array()[...] = output["arr_%i"%i][...]

    def export_json(self):
        return {"updates":[update.export_json() for update in self.updates]}

    def import_json(self, json):
        for i,json_update in enumerate(json["updates"]):
            self.updates[i].import_json(json_update)
