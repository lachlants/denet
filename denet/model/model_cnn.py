import os
import sys
import numpy
import base64
import json
import io
import getpass
import math
import random
import time
import theano
import theano.tensor as tensor

import denet.common as common
import denet.common.logging as logging
import denet.layer
from denet.layer import InitialLayer, IdentityLayer
from denet.layer.layer_types import layer_types

#enable for debugging (Warning slow)
debug_train = False

#load from raw JSON
def load_from_json(json_obj, batch_size=32, layer_range=None):
    model = ModelCNN()
    model.batch_size = batch_size
    model.import_json(json_obj, layer_range)
    return model

#load model from gz JSON file
def load_from_file(fname, batch_size=32, layer_range=None):
    t = time.time()
    logging.info("Loading model from %s" % fname)
    model = load_from_json(common.json_from_gz(fname), batch_size, layer_range)
    model.fname = fname
    logging.verbose("Loading model took %.2f sec"%(time.time()-t))
    return model

#save model to zipped JSON file
def save_to_file(model, fname, compresslevel=9):
    logging.info("Saving model to %s" % fname)
    t = time.time()
    common.json_to_gz(fname, model.export_json(), compresslevel)
    logging.verbose("Saving model took %.2f sec"%(time.time()-t))

def initialize(args, data_shape, class_labels, class_num):

    cudnn_info=(theano.config.dnn.conv.algo_fwd, theano.config.dnn.conv.algo_bwd_data, theano.config.dnn.conv.algo_bwd_filter)
    logging.info("Using theano version:", theano.__version__, "(cudnn fwd=%s,bwd data=%s,bwd filter=%s)"%cudnn_info)
    if args.model is None:

        #construct convolutional model
        logging.info("Building convolutional model (%i classes)..."%class_num)
        model = ModelCNN()
        model.batch_size = args.batch_size
        model.class_labels = class_labels
        model.class_num = class_num

        #allow padding to be specified in border mode
        try:
            n = int(args.border_mode)
            border_mode = (n,n)
        except ValueError:
            border_mode = args.border_mode

        model.build(args.model_desc, data_shape, args.activation, border_mode, list(args.weight_init))
    else:
        model = load_from_file(args.model, args.batch_size)
        model.class_labels = class_labels
        model.class_num = class_num
        assert data_shape == model.data_shape, "Mismatching data shapes in .mdl and data: " + str(data_shape) + "!="  + str(model.data_shape)

    model.skip_layer_updates = args.skip_layer_updates
    if len(model.skip_layer_updates) > 0:
        logging.info("Skipping layer updates:", model.skip_layer_updates)

    return model

#
class ModelCNN():

    def __init__(self):
        super().__init__()

        self.batch_size = 0
        self.iteration = 0
        self.class_labels = None
        self.data_shape = None
        self.class_num = 0
        self.rng_seed = random.randint(1,9999)
        denet.layer.set_rng_seed(self.rng_seed)

        #training parameters
        self.gradient_clip = 0.0
        self.skip_layer_updates = []
        self.bias_decay = False
        self.layers=[]
        self.distort_mode = []
        self.func = {}

        #input data
        self.input = tensor.tensor4("input")

    #input image shape
    def get_input_shape(self):
        assert not self.data_shape is None, "Data shape hasn't been set!"
        return tuple([self.batch_size] + list(self.data_shape))

    #model output shape
    def get_output_shape(self):
        return self.layers[-1].output_shape

    #get total number of parameters in model
    def get_parameter_num(self):
        n=0
        for layer in self.layers:
            for param in layer.params():
                n += param.get_value(borrow=True, return_internal_type=True).size
        return n;

    #
    def build_layer(self, layer_desc, layers, activation, border_mode, wb):

        p_start = layer_desc.find("[")
        p_end = layer_desc.find("]")
        layer_params = {"classNum":self.class_num, "activation":activation, "borderMode":border_mode, "wb": wb}
        if p_start > 0 and p_end > p_start:
            layer_type = layer_desc[:p_start]
            for i,p in enumerate(layer_desc[(p_start+1):p_end].split(",")):
                layer_params[i] = common.convert_num(p)
        else:
            layer_type = layer_desc

        t_index = layer_type.find(".")
        if t_index > 0:
            layer_tags = layer_type[(t_index+1):]
            layer_type = layer_type[:t_index]
        else:
            layer_tags = ""

        for layer in layer_types:
            if layer.parse_desc(layers, layer_type, layer_tags, layer_params):
                return

        raise Exception("Invalid layer - type: ", layer_type, "tags:", layer_tags, "params:", layer_params)

    #build using model parameter string
    def build(self, model_desc, data_shape, activation = "relu", border_mode="valid", weight_init="he-forward"):

        self.model_desc = " ".join(model_desc)
        self.data_shape = data_shape
        self.layers = [InitialLayer(self.input, self.get_input_shape())]
        for i,layer_desc in enumerate(model_desc):
            wb = weight_init[min(len(weight_init)-1, i)]
            self.build_layer(layer_desc, self.layers, activation, border_mode, wb)

        logging.verbose("Number of parameters in model: %d" % self.get_parameter_num())

    def export_json(self):
        json_layers=[]
        for index in range(1, len(self.layers)):
            json_layers.append(self.layers[index].export_json())
        
        from time import gmtime, strftime
        json_obj={"classifierType" : "CNN",
                  "classLabels" : self.class_labels,
                  "classNum" : self.class_num,
                  "dataShape": self.data_shape,
                  "date" : strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                  "user" : getpass.getuser()}

        json_obj.update({"version" : 3, "layers" : json_layers})
        return json_obj

    def import_json(self, json_obj, layer_range=None):

        self.func={}

        #check if old JSON format
        if json_obj.get("version", 0) == 0:
            raise Exception("Old format model file detected, no compatibility!")
            return

        self.class_labels = json_obj["classLabels"]

        if "imageSize" in json_obj and "imageMode" in json_obj:
            width = json_obj["imageSize"][0]
            height = json_obj["imageSize"][1]
            image_mode = json_obj.get("imageMode", "RGB")
            self.data_shape = ({"RGB":3, "L":1}[image_mode], width, height)
        elif "dataShape" in json_obj:
            self.data_shape = tuple(json_obj["dataShape"])
        else:
            assert False, "Bad mdl file, Cannot determine input data shape!"

        assert json_obj.get("imageBorder", 0) == 0

        self.class_num =json_obj.get("classNum", len(self.class_labels))

        #load layers
        self.layers = denet.layer.import_json(json_obj["layers"], self.input, self.get_input_shape(), layer_range)

        logging.info("Number of parameters in model: %d" % self.get_parameter_num())

    def build_train_func(self, solver_mode="sgd", cost_factors=[], use_acc_mode=False, skip_build=False):

        #arguments to function
        logging.info("Building training functions - solver: %s, use_acc_mode: %s"%(solver_mode, use_acc_mode))
        iteration = tensor.fscalar()
        learn_rate = tensor.fscalar()
        momentum = tensor.fvector()
        decay = tensor.fscalar()

        #find costs
        self.yt = []
        self.cost_list = []
        self.cost_layers = [] 
        self.cost_layer_names = []
        for layer in self.layers:
            yt_index = tensor.lvector("target index %i"%len(self.cost_layers))
            yt_value = tensor.fvector("target value %i"%len(self.cost_layers))
            cost = layer.cost(yt_index, yt_value)
            if not cost is None:
                self.yt += [yt_index, yt_value]
                self.cost_list.append(cost)
                self.cost_layers.append(layer)
                self.cost_layer_names.append(layer.type_name)
                
        self.cost_factors = [1.0]*len(self.cost_list) if len(cost_factors) == 0 else cost_factors
        assert len(self.cost_factors) == len(self.cost_list), "Different number of cost factors (%i) and cost layers (%i)"%(len(self.cost_factors), len(self.cost_layers))
        logging.info("Found %i costs in model:"%len(self.cost_layers), list(zip(self.cost_layer_names, self.cost_factors)))

        self.train_cost = tensor.as_tensor_variable(0)
        for i,cost in enumerate(self.cost_list):
            self.train_cost += self.cost_factors[i]*cost

        if self.gradient_clip > 0.0:
            logging.info("Clipping gradient to [%f,%f]"%(-self.gradient_clip, self.gradient_clip))
            self.train_cost = theano.gradient.grad_clip(self.train_cost, -self.gradient_clip, self.gradient_clip)

        #find split points
        split_points=[0]
        self.use_split_mode = False
        for index, layer in enumerate(self.layers):
            if layer.has_split:
                self.use_split_mode = True
                split_points.append(index)
        split_points.append(len(self.layers))

        if self.use_split_mode:
            logging.verbose("Using split mode with split points:", split_points)
            self.func["train_fwd"]=[]
            self.func["train_bwd"]=[]

        self.updates=[]
        for sp in range(len(split_points)-1):

            logging.info("Building training functions for layers %i-%i"%(split_points[sp], split_points[sp+1]))

            split_start = self.layers[split_points[sp]] if sp > 0 else None
            split_end = self.layers[split_points[sp+1]] if (sp+2) < len(split_points) else None
            split_cost = self.train_cost if split_end is None else None
            split_layers = []
            for i,layer in enumerate(self.layers):
                if (i > split_points[sp]) and (i < split_points[sp+1]):
                    split_layers.append(layer)

            #determine known_grads provided by previous backward passes
            from collections import OrderedDict
            split_known_grads=OrderedDict()
            for i in range(sp+1, len(split_points)-1):
                split_known_grads.update(self.layers[split_points[i]].split_known_grads())

            if len(split_known_grads) == 0:
                split_known_grads=None

            # print(split_known_grads)
            # print(split_known_grads)
            # print(sp+1, len(split_points)-1)

            #
            def get_sgd_updates(p, g):
                m = theano.shared(numpy.zeros(p.shape.eval(), dtype=theano.config.floatX), broadcastable=p.broadcastable, borrow=True)
                rho = tensor.switch(tensor.gt(iteration, 0), momentum[0], 0.0)
                m_update = rho*m + (1.0 - rho)*g
                p_update = p - learn_rate*m_update
                return [(p, p_update), (m, m_update)]

            def get_torch_updates(p, g):
                m = theano.shared(numpy.zeros(p.shape.eval(), dtype=theano.config.floatX), broadcastable=p.broadcastable, borrow=True)
                rho = tensor.switch(tensor.gt(iteration, 0), momentum[0], 0.0)
                m_update = rho*m + g
                p_update = p - learn_rate*(g + momentum[0]*m_update)
                return [(p, p_update), (m, m_update)]

            def get_adam_updates(p, g):
                eps = 1e-8
                m = theano.shared(numpy.zeros(p.shape.eval(), dtype=theano.config.floatX), broadcastable=p.broadcastable, borrow=True)
                v = theano.shared(numpy.zeros(p.shape.eval(), dtype=theano.config.floatX), broadcastable=p.broadcastable, borrow=True)
                m_update = momentum[0]*m + (1.0 - momentum[0])*g
                v_update = momentum[1]*v + (1.0 - momentum[1])*(g * g)
                m_hat = m_update / (1.0 - tensor.pow(momentum[0], iteration + 1))
                v_hat = v_update / (1.0 - tensor.pow(momentum[1], iteration + 1))
                p_update = p - learn_rate*m_hat/(tensor.sqrt(v_hat) + eps)
                return [(p, p_update), (m, m_update), (v, v_update)]

            #append parameter updates
            params=[]
            params_decay=[]
            for layer in split_layers:
                params += layer.weights()
                params_decay += [True]*len(layer.weights())
                params += layer.biases()
                params_decay += [False]*len(layer.biases())

            #build updates
            print("known grads:", split_known_grads)
            grads = tensor.grad(split_cost, params, known_grads=split_known_grads)
            solver_updates=[]
            for p, g, p_decay in zip(params, grads, params_decay):

                #add L2 weight decay if needed
                if p_decay or self.bias_decay:
                    g += decay*p

                if solver_mode == "adam":
                    solver_updates += get_adam_updates(p, g)
                elif solver_mode == "torch" or solver_mode == "nesterov":
                    solver_updates += get_torch_updates(p, g)
                else:
                    solver_updates += get_sgd_updates(p, g)

            #append per layer updates
            local_updates = solver_updates + sum([layer.updates(self.train_cost) for layer in split_layers],[])

            #all updates
            self.updates += local_updates
            
            #skipping actual theano function building (if you just want updates, etc)
            if skip_build:
                continue

            global debug_train
            if debug_train:
                logging.warning("WARNING: Debug mode is active!")
                from theano.compile.nanguardmode import NanGuardMode
                debug_mode = theano.compile.MonitorMode(post_func=debug_detect_errors)
            else:
                debug_mode = None

            if self.use_split_mode:

                if not split_end is None:
                    updates = sum([layer.split_forward() for layer in split_layers], [])
                    updates += split_end.split_forward()

                    print("fwd updates:", updates)
                    f = theano.function([self.input], [], updates=updates, givens=[(denet.layer.get_train(), tensor.cast(1, 'int8'))],
                                        on_unused_input='ignore', mode = debug_mode)
                    self.func["train_fwd"].append(f)

                outputs = ([self.train_cost] + self.cost_list) if split_end is None else []
                updates = sum([layer.split_backward(split_cost, split_known_grads) for layer in split_layers], [])
                if not split_start is None:
                    updates += split_start.split_backward(split_cost, split_known_grads)

                print("bwd updates:", updates)
                updates += local_updates
                f = theano.function([denet.layer.get_epoch(), iteration, learn_rate, momentum, decay, self.input] + self.yt, 
                                    outputs, updates=updates, givens=[(denet.layer.get_train(), tensor.cast(1, 'int8'))],
                                    on_unused_input='ignore', mode = debug_mode)
                self.func["train_bwd"].insert(0, f)
                
            elif use_acc_mode:
                acc_counter = theano.shared(numpy.array(0, dtype=theano.config.floatX))
                begin_updates=[(acc_counter, tensor.zeros_like(acc_counter))]
                step_updates=[(acc_counter, acc_counter+1)]
                end_updates=[]
                self.acc_params=[]
                for p_dest, p_src in self.updates:
                    p_acc = theano.shared(numpy.zeros(p_dest.shape.eval(), dtype=theano.config.floatX), broadcastable=p_dest.broadcastable, borrow=True)
                    begin_updates.append((p_acc, tensor.zeros_like(p_acc)))
                    step_updates.append((p_acc, p_acc + p_src))
                    end_updates.append((p_dest, p_acc / acc_counter))
                    self.acc_params.append(p_acc)

                logging.info("Constructing parameter accumulate update functions (solver=%s)"%solver_mode)
                self.func["train_begin"] = theano.function([],[], updates=begin_updates)
                self.func["train_step"] = theano.function([denet.layer.get_epoch(), iteration, learn_rate, momentum, decay, self.input] + self.yt, [self.train_cost] + self.cost_list,
                                                 updates= step_updates, givens=[(denet.layer.get_train(), tensor.cast(1, 'int8'))],
                                                 on_unused_input='ignore', allow_input_downcast=True, mode = debug_mode)
                self.func["train_end"] = theano.function([],[], updates=end_updates)
            else:
                logging.info("Constructing parameter update function (solver=%s)"%solver_mode)
            
                #making 
                f_input = theano.In(self.input, borrow=True)
                f_yt = [theano.In(yt, borrow=True) for yt in self.yt]
                self.func["train_step"] = theano.function([denet.layer.get_epoch(), iteration, learn_rate, momentum, decay, f_input] + f_yt, [self.train_cost] + self.cost_list,
                                                        updates=self.updates, givens=[(denet.layer.get_train(), tensor.cast(1, 'int8'))],
                                                        on_unused_input='ignore', allow_input_downcast=True, mode = debug_mode)

                logging.verbose("Exporting graph...")
                with open("graph.txt", "w") as f:
                    theano.printing.debugprint(self.func["train_step"], file=f, print_type=True)
            
    def train_step(self, data_x, data_m, epoch, it, learning_rate, momentum, decay):
        #assert "train" in self.func, "Call build_train_func() before calling train_step()"
        denet.layer.get_iteration().set_value(it)
        momentum=numpy.array(momentum, dtype=numpy.float32)
        
        #for split mode explicitly perform bwd/fwd propagation
        if self.use_split_mode:

            #only get_targets after train_fwd has been performed
            targets = []
            fwd_index=0
            for layer in self.layers:
                if layer.has_split:
                    logging.verbose("Forward prop %i"%fwd_index)
                    self.func["train_fwd"][fwd_index](data_x)
                    fwd_index += 1

                target = layer.get_target(self, data_x, data_m)
                if not target is None:
                    print(layer.type_name)
                    targets += target

            # print(targets)

            logging.verbose("Backward prop %i"%0)
            costs = self.func["train_bwd"][0](epoch, it, learning_rate, momentum, decay, data_x, *targets)
            for i,f_bwd in enumerate(self.func["train_bwd"][1:]):
                logging.verbose("Backward prop %i"%(i+1))
                f_bwd(epoch, it, learning_rate, momentum, decay, data_x, *targets)
        else:
            targets=[]
            for layer in self.layers:
                target = layer.get_target(self, data_x, data_m)
                if not target is None:
                    targets += target

            costs = self.func["train_step"](epoch, it, learning_rate, momentum, decay, data_x, *targets)

        return costs[0], costs[1:] 

    def train_epoch(self, dataset, epoch, learning_rate, momentum=[0,1,0], decay=0.0, solver_mode="sgd"):
        
        #train over batches (assume dataset size is mulitple of batch_size!)
        logging.info("Evaluating training function")
        dataset_x, dataset_m, dataset_size = dataset.export(self.batch_size)
        index_num = math.ceil(dataset_size / self.batch_size)
        total_cost=0
        for index in range(index_num):

            #upload data to GPU and perform train step
            timer = common.Timer()
            data_x = dataset_x[index * self.batch_size : (index + 1) * self.batch_size]
            data_m = dataset_m[index * self.batch_size : (index + 1) * self.batch_size]
            cost, _ = self.train_step(data_x, data_m, epoch, self.iteration, learaning_rate, momentum, decay)

            #watch out for GPU's randomly producing NaN!
            if math.isnan(cost):
                raise Exception("ERROR: Cost is NaN")

            logging.verbose("Batch %i.%i - iteration: %i cost:"%(epoch, index*self.batch_size, self.iteration), cost, "took: %i ms"%timer.current_ms())
            total_cost += cost
            self.iteration += 1

        return total_cost

    #returns a numpy array of size (len(dataset), class_num) with each element describing the probability
    def predict_output_step(self, data_x):
        if not "predict" in self.func:
            logging.info("Building predict function")
            self.func["predict"] = theano.function([self.input], self.layers[-1].output, givens=[(denet.layer.get_train(), tensor.cast(0, 'int8'))], on_unused_input='ignore')

        return self.func["predict"](data_x)

    #returns a numpy array of size (len(dataset), class_num) with each element describing the probability
    def predict_output(self, dataset):
        dataset_x, dataset_y, dataset_size = dataset.export(self.batch_size)

        #dummy call to build function
        self.predict_output_step(dataset_x[:self.batch_size])

        #evaluate function
        timer = common.Timer()
        n = math.ceil(dataset_size / self.batch_size)
        pr=[]
        for index in range(n):
            data_x = dataset_x[index*self.batch_size:(index+1)*self.batch_size]
            pr_batch = self.predict_output_step(data_x)
            pr.append(pr_batch)
        pr = numpy.concatenate(pr, axis=0)

        logging.verbose("Prediction took %.3f sec for %i samples"%(timer.current(), pr.shape[0]))

        #crop dummy data
        if (dataset_size%self.batch_size) != 0:
            s = [dataset_size] + list(pr.shape[1:])
            pr.resize(tuple(s), refcheck=False)

        return pr

    #assigns a numeric label to each sample in dataset (i.e. finds argmax of probabilities)
    def predict_label(self, dataset):
        pr = self.predict_output(dataset)
        assert pr.ndim == 2
        predict = [numpy.argmax(pr[i, ...]) for i in range(pr.shape[0])]
        return predict

    #result result of tensor
    def predict_custom(self, dataset, output_func, output_shape):

        #upload data to GPU
        dataset_x, dataset_y, dataset_size = dataset.export(self.batch_size)

        #evaluate function
        n = math.ceil(dataset_size / self.batch_size)
        shape = tuple([n*output_shape[0]] + list(output_shape)[1:])
        result = numpy.zeros(shape, dtype=numpy.float32)
        for index in range(n):
            i0 = index*self.batch_size
            i1 = (index+1)*self.batch_size
            data_x = dataset_x[i0:i1]
            data_y = output_func(data_x)

            i0 = index*output_shape[0]
            i1 = (index+1)*output_shape[0]
            result[i0:i1, ...] = data_y[...]

        #crop dummy data
        if (dataset_size % self.batch_size) != 0:
            s = list(result.shape)
            s[0] = dataset_size
            result.resize(tuple(s), refcheck=False)

        return result

def debug_denet_export_targets(model, data_x, data_m, targets):
    logging.debug("DEBUGGING! Exporting targets")
    class_labels_inv = { v:k for k,v in model.class_labels.items()}
    for b in range(model.batch_size):

        #export ground truth
        for cls in set(data_m[b]["class"]):
            objs = []
            for obj_cls,obj in zip(data_m[b]["class"], data_m[b]["objs"]):
                if obj_cls == cls:
                    objs.append(obj)
            common.export_activation_rgb("%06i_gt_%s.png"%(b, class_labels_inv[cls]), data_x[b,:,:,:], objs)

        #export targets
        for index,layer in enumerate(model.cost_layers):

            yt_index = targets[index*2+0]
            yt_value = targets[index*2+1]
            if layer.type_name == "denet-corner":
                corner_pr, = common.ndarray_unpack(yt_value, [layer.corner_shape])
                common.export_activation("%06i_l%i_corner.png"%(b,index), corner_pr[b,1,:,:,:])

            elif layer.type_name == "denet-detect":
                det_pr, = common.ndarray_unpack(yt_value, [layer.det_shape])
                for i, sample in enumerate(layer.sparse_layer.sample_bboxs[b]):
                    sample_i = i % layer.sparse_layer.sample_num
                    sample_j = i // layer.sparse_layer.sample_num
                    logging.debug("%i - sample:"%b, (sample_i, sample_j), "bbox:", (int(sample[2][0]*model.width), int(sample[2][1]*model.height), int(sample[2][2]*model.width), 
                                                                                    int(sample[2][3]*model.height)), "pr:", sample[1], "corner:", sample[0])

                common.export_activation("%06i_l%i_det.png"%(b,index), det_pr[b,:,:,:])
