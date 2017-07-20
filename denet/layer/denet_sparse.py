import sys
import os
import theano
import theano.tensor as tensor
import numpy
import math
import time
import random
import threading

import denet.common as common
import denet.common.logging as logging
import denet.common.theano_util as theano_util
from denet.layer import AbstractLayer, InitialLayer, get_train
from denet.layer.denet_sparse_op import DeNetSparseOp

#load C CODE extensions
c_code = common.import_c(os.path.join(os.path.dirname(__file__), "denet_sparse.cc"))
assert not c_code is None
c_code.init_logging(theano.config.device + "_sparse_cc.log")

#enable theano corner function profiling
profile=False

#define layer
class DeNetSparseLayer(AbstractLayer):
    type_name = "denet-sparse"
    def __init__(self, layers, grid_size=3, sample_num=16, corner_threshold=0.01, random_sample=0.0, 
                 local_max = 0, nms_threshold=0.7, sample_gt=True, version="v2", json_param={}):

        super().__init__(layer_index=len(layers))

        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape
        self.batch_size = self.input_shape[0]
        self.model_input = layers[0].output
        
        self.grid_size = json_param.get("gridSize", grid_size)
        self.sample_num = json_param.get("sampleNum", sample_num)
        self.sample_gt = json_param.get("sampleGT", sample_gt)
        self.corner_threshold = json_param.get("cornerThreshold", corner_threshold)
        self.nms_threshold = json_param.get("nmsThreshold", nms_threshold)
        self.random_sample = json_param.get("randomSample", random_sample)
        self.local_max = json_param.get("localMax", local_max)
        self.version = json_param.get("version", version)
        
        self.corner_max = 1024
        self.thread_num = self.batch_size
        self.sample_count = self.sample_num*self.sample_num

        #find corner layer
        self.corner_func = None
        self.corner_layer = common.find_layers(layers, "denet-corner", True)
        assert not self.corner_layer is None, "denet-corner layer required before spare layer!"

        #sampling bounding boxs
        self.sample_bbox_list = []
        self.sample_bbox = theano.shared(value=numpy.zeros((self.batch_size, self.sample_num, self.sample_num, 4), dtype=numpy.float32))

        self.output_feat = self.grid_size*self.grid_size*self.corner_layer.sample_shape[1]+2
        self.output_shape = (self.batch_size, self.output_feat, self.sample_num, self.sample_num)

        #Use optimized op! 
        sample_input = tensor.switch(get_train(), self.corner_layer.sample, self.corner_layer.sample_shared)
        self.use_optimized = theano.sandbox.cuda.cuda_enabled
        if self.use_optimized:
            self.output = DeNetSparseOp(self.grid_size)(sample_input, self.sample_bbox)
        else:

            sample_w = self.sample_bbox[:,:,:,2] - self.sample_bbox[:,:,:,0]
            sample_h = self.sample_bbox[:,:,:,3] - self.sample_bbox[:,:,:,1]
            sample_x = self.sample_bbox[:,:,:,0,None] + tensor.arange(self.grid_size, dtype='float32')[None,None,None,:]*sample_w[:,:,:,None] / (self.grid_size-1)
            sample_y = self.sample_bbox[:,:,:,1,None] + tensor.arange(self.grid_size, dtype='float32')[None,None,None,:]*sample_h[:,:,:,None] / (self.grid_size-1)
            batch_range = tensor.arange(self.batch_size, dtype='int64')[:,None,None,None,None]
            grid_range = tensor.arange(self.grid_size*self.grid_size, dtype='int64').reshape((self.grid_size, self.grid_size))[None,None,None,:,:]

            #extract samples
            b,f,h,w = self.corner_layer.sample_shape

            #get sample position in feature map (b,sj,si,gy/gx)
            sample_xf = tensor.maximum(0, tensor.minimum(sample_x*w, w - 1))
            sample_yf = tensor.maximum(0, tensor.minimum(sample_y*h, h - 1))
            syf = tensor.cast(sample_yf.round(), 'int64')
            sxf = tensor.cast(sample_xf.round(), 'int64')
            byxc = batch_range*h*w + syf[:,:,:,:,None]*w + sxf[:,:,:,None,:]

            #extract sample
            sample = sample_input.dimshuffle((1,0,2,3)).flatten(2)
            sample = sample[:, byxc.flatten()]
            sample = sample.reshape((f, b, self.sample_num, self.sample_num, self.grid_size*self.grid_size))
            sample = sample.dimshuffle((1,2,3,4,0))
            sample = sample.reshape((b, self.sample_num, self.sample_num, self.grid_size*self.grid_size*f))
            sample = sample.dimshuffle((0,3,1,2))

            #add width / height
            self.output = tensor.concatenate([sample, sample_h[:,None,:,:], sample_w[:,None,:,:]], axis=1)

        global c_code
        logging.verbose("Adding", self)

    #parse layer desciptor
    def parse_desc(layers, name, tags, params):
        if name != "DNS":
            return False

        grid_size = params.get(0, 3)
        sample_num = params.get(1, 4)
        corner_threshold = params.get(2, 0.01)
        random_sample = params.get(3, 0.1)
        local_max = params.get(4, 0)
        nms_threshold = params.get(5, 1.0)
        sample_gt = not "G" in tags
        layers.append(DeNetSparseLayer(layers, grid_size, sample_num, corner_threshold, random_sample, local_max, nms_threshold, sample_gt))
        return True

    #run corner detector to obtain sample bboxs
    def get_samples(self, data_x, train=False, store_shared=False):

        global profile
        if self.corner_func is None:
            logging.verbose("Building corner function - store samples:", store_shared, "train:", train)
            updates = [(self.corner_layer.sample_shared, self.corner_layer.sample)] if store_shared else []
            self.corner_func = theano.function([self.model_input], self.corner_layer.corner_pr, updates=updates, profile=profile, 
                                               givens=[(get_train(), tensor.cast(int(train), 'int8'))], on_unused_input='ignore')

        #find corners
        timer = common.Timer()
        logging.debug("Running corner function")
        corner_pr = self.corner_func(data_x)
            
        if profile:
            logging.debug("Profiling corner function")
            theano_util.profile(self.corner_func, 10, data_x)
            theano_util.export_graph("./corner.graph", self.corner_func)
            logging.debug("Done")
            exit(0)
  
        #build sampling bounding boxs
        timer.mark()
        logging.debug("Build samples (%i threads)"%self.thread_num)
        samples = c_code.build_samples(self.thread_num, corner_pr, self.corner_threshold, self.sample_num, self.corner_max, self.local_max, self.nms_threshold)

        timer.mark()
        logging.verbose("Took %i ms to get_samples (%i model, %i build, %i max corners) "%(timer.current_ms(), timer.delta_ms(0), timer.delta_ms(1), self.corner_max))
        return samples

    #extract bboxs
    def get_bbox_array(self, sample_bboxs):
        timer = common.Timer()
        bboxs = numpy.zeros((self.batch_size, self.sample_num, self.sample_num, 4), dtype = numpy.float32)
        c_code.build_bbox_array(sample_bboxs, bboxs)
        logging.debug("Took %i ms to get_bbox_array"%timer.current_ms())
        return bboxs

    def set_samples(self, sample_bboxs):
        timer = common.Timer()
        bboxs = self.get_bbox_array(sample_bboxs)
        self.sample_bbox.set_value(bboxs)
        self.sample_bbox_list = sample_bboxs
        logging.debug("Took %i ms to set_samples"%timer.current_ms())
        return bboxs

    #dummy get_target
    def get_target(self, model, data_x, metas):

        sample_bboxs = self.get_samples(data_x, train=True)

        total_cover=0
        total_bbox=0
        for b, meta in enumerate(metas):
                
            #compute groundtruth coverage
            cover = 0
            for meta_bbox in meta["bbox"]:
                for _, sample_bbox in sample_bboxs[b]:
                    if common.overlap_iou(meta_bbox, sample_bbox) > 0.5:
                        cover += 1
                        break

            logging.verbose("%i: corner detector found %i samples (%i/%i coverage)"%(b, len(sample_bboxs[b]), cover, len(meta["bbox"])))
            total_cover += cover
            total_bbox += len(meta["bbox"])

            n = self.sample_count - math.floor(self.random_sample*self.sample_count)
            if len(sample_bboxs[b]) > n:
                # logging.verbose("%i: removing %i samples to make room for random samples"%(b, len(sample_bboxs[b]) - n))
                sample_bboxs[b] = random.sample(sample_bboxs[b], n)
            
            #add random samples if bbox detector produces too few
            while len(sample_bboxs[b]) < self.sample_count:
                x0 = random.uniform(0.0,1.0)
                y0 = random.uniform(0.0,1.0)
                x1 = random.uniform(x0,1.0)
                y1 = random.uniform(y0,1.0)
                bbox=(x0,y0,x1,y1)
                sample_bboxs[b].append((0.0, bbox))
        
            #insert groundtruth
            if self.sample_gt:
                for index,bbox in enumerate(meta["bbox"]):
                    sample_bboxs[b][-(index+1)] = (1.0, bbox)

        logging.verbose("Overall %i/%i (%.2f%%) coverage"%(total_cover, total_bbox, 100.0*total_cover / total_bbox))

        self.set_samples(sample_bboxs)
        return None

    def export_json(self):
        json=super().export_json()
        json.update({"gridSize" : self.grid_size,
                     "sampleNum" : self.sample_num,
                     "sampleGT" : self.sample_gt,
                     "localMax" : self.local_max,
                     "cornerThreshold" : self.corner_threshold,
                     "randomSample" : self.random_sample,
                     "nmsThreshold" : self.nms_threshold,
                     "version" : self.version})
        return json


#debugging
if __name__ == '__main__':
        
    logging.init()
    from denet.layer.denet_sparse_op import DeNetSparseOp
    import time

    batch_size=32
    feature_num=64
    height=32
    width=32
    sample_num=24
    grid_size=7

    numpy.random.seed(1)
    random.seed(1)

    x = numpy.random.uniform(-5, 5, (batch_size, feature_num, height, width)).astype(numpy.float32)
    y = numpy.zeros((batch_size, sample_num, sample_num, 4)).astype(numpy.float32)
    for b in range(batch_size):
        for index in range(sample_num*sample_num):
            j = index // sample_num
            i = index % sample_num
            y[b,j,i,0] = random.uniform(0.0, 1.0)
            y[b,j,i,1] = random.uniform(0.0, 1.0)
            y[b,j,i,2] = random.uniform(y[b,j,i,0], 1.0)
            y[b,j,i,3] = random.uniform(y[b,j,i,1], 1.0)

    # y = numpy.zeros((batch_size, sample_num, sample_num, 4)).astype(numpy.float32)
    # y[:,:,:,0] = 0.0
    # y[:,:,:,1] = 0.0
    # y[:,:,:,2] = 1.0
    # y[:,:,:,3] = 1.0

    print(x.shape)
    print(y.shape)
    
    fmap = tensor.tensor4()
    fmap_shape = x.shape

    class CornerDummy:
        type_name = "denet-corner"
        def __init__(self, x, x_shape):
            self.output = self.sample = x
            self.output_shape = self.sample_shape = x_shape
            self.sample_shared = theano.shared(value=numpy.zeros(x_shape, dtype=numpy.float32))
            self.sample_feat = x_shape[1]

    sparse_layer = DeNetSparseLayer([CornerDummy(fmap, fmap_shape)], grid_size, sample_num)
    sparse_layer.sample_bbox.set_value(y)
    cost = sparse_layer.output.sum()


    print("Building function")
    f_new = theano.function([fmap], tensor.grad(cost, fmap), givens=[(get_train(), tensor.cast(0, 'int8'))],)
    f_old = theano.function([fmap], tensor.grad(cost, fmap), givens=[(get_train(), tensor.cast(1, 'int8'))],)
    r_new = f_new(x)
    r_old = f_old(x)

    print("------- NEW -------")
    print(r_new.shape)
    print(r_new[:,0,:,:])
    print("------- OLD -------")
    print(r_old.shape)
    print(r_old[:,0,:,:])
