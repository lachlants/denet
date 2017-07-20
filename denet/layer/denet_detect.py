import theano
import theano.tensor as tensor
import numpy
import math
import random
import scipy
import time
import os

import denet.common as common
import denet.common.logging as logging
import denet.common.theano_util as theano_util

from denet.layer import AbstractLayer, InitialLayer, get_train
from denet.layer.convolution import ConvLayer

#used for timing info
detect_time=0 
detect_num=0

#import c_code optimizations
c_code = common.import_c(os.path.join(os.path.dirname(__file__), "denet_detect.cc"))
assert not c_code is None

class DeNetDetectLayer(AbstractLayer):

    type_name = "denet-detect"
    def __init__(self, layers, class_num=10, overlap_threshold = 0.5, cost_factor=1.0, bbox_factor=0.0, json_param={}):

        super().__init__(layer_index=len(layers))

        #passthrough layer
        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape
        self.output = layers[-1].output
        self.output_shape = layers[-1].output_shape

        self.cost_factor = json_param.get("costFactor", cost_factor)
        self.bbox_factor = json_param.get("bboxFactor", bbox_factor)
        self.class_num = json_param.get("classNum", class_num)
        self.overlap_threshold = json_param.get("overlapThreshold", overlap_threshold)
        
        #find sparse / instcount / compare layer
        self.sparse_layer = common.find_layers(layers, "denet-sparse", False)
        assert self.sparse_layer != None, "Error: Requires denet-sparse layer to be specified before denet-detect layer!"

        self.use_bbox_reg = (self.bbox_factor > 0.0)
        self.batch_size = self.sparse_layer.batch_size
        self.sample_num = self.sparse_layer.sample_num
        self.null_class = self.class_num

        #
        s0 = self.class_num+1
        s1 = 4 if self.use_bbox_reg else 0
        self.layers = [ConvLayer([InitialLayer(self.input, self.input_shape)], (s0+s1, self.input_shape[1], 1, 1), (1,1), True, "valid", 0.0)]

        #class assignment log(Pr({c,null} | sample_j, sample_i))
        self.det_shape = (self.batch_size, s0, self.sample_num, self.sample_num)
        self.det_lh = self.layers[-1].output[:, :s0 ,...]
        self.det_pr = theano_util.log_softmax(self.det_lh, axis=[1])

        #bbox regression
        if self.use_bbox_reg:
            self.bbox_shape = (self.batch_size, s1, self.sample_num, self.sample_num)
            self.bbox_reg = self.layers[-1].output[:, s0:(s0+s1), ...]
        
        self.detect_func = None
        self.nms_func = None

        logging.verbose("Adding", self)

    #parse layer desciptor
    def parse_desc(layers, name, tags, params):
        if name != "DND":
            return False

        overlap_threshold = params.get(0, 0.5)
        cost_factor = params.get(1, 1.0)
        bbox_factor = params.get(2, 0.0)
        layers.append(DeNetDetectLayer(layers, params.get("classNum"), overlap_threshold, cost_factor, bbox_factor))
        return True

    def import_json(self, json_param):
        super().import_json(json_param)

        #backward compatibility
        if "conv" in json_param:
            self.layers[0].import_json(json_param["conv"])

    def export_json(self):
        json=super().export_json()
        json.update({"costFactor" : self.cost_factor, 
                     "bboxFactor" : self.bbox_factor, 
                     "classNum": self.class_num, 
                     "overlapThreshold": self.overlap_threshold})
        return json

    def get_target(self, model, samples, metas):

        timer = common.Timer()

        #build sample
        det_pr = numpy.zeros(self.det_shape, dtype=numpy.float32)
        det_pr[:, self.null_class, ...] = 1.0
        
        if self.use_bbox_reg:
            bbox_valid = numpy.zeros((self.batch_size, self.sample_num, self.sample_num), dtype=numpy.float32)
            bbox_reg = numpy.ones((self.batch_size, 8, self.sample_num, self.sample_num), dtype=numpy.float32)

        for b, meta in enumerate(metas):

            samples = [bbox for _,bbox in self.sparse_layer.sample_bbox_list[b]]
            if len(meta["bbox"]) > 0 and len(samples) > 0:
                overlap = theano_util.get_overlap_iou(meta["bbox"], samples)
                bbox_indexs, sample_indexs = numpy.where(overlap > self.overlap_threshold)
                for obj,index in zip(bbox_indexs.tolist(), sample_indexs.tolist()):
                    sample_i = index % self.sparse_layer.sample_num
                    sample_j = index // self.sparse_layer.sample_num
                    sample_cls = meta["class"][obj]
                    sample_bbox = samples[index]
                    det_pr[b, sample_cls, sample_j, sample_i] = 1.0
                    det_pr[b, self.null_class, sample_j, sample_i] = 0.0

                if self.use_bbox_reg:
                    overlap_max = overlap.argmax(axis=0)
                    for index in range(len(samples)):
                        obj = overlap_max[index]
                        if overlap[obj, index] <= self.overlap_threshold:
                            continue

                        sample = samples[index]
                        target = meta["bbox"][obj]
                        sample_i = index % self.sparse_layer.sample_num
                        sample_j = index // self.sparse_layer.sample_num
                        bbox_valid[b, sample_j, sample_i] = 1.0
                        bbox_reg[b, 0, sample_j, sample_i] = 0.5*(target[0]+target[2])
                        bbox_reg[b, 1, sample_j, sample_i] = 0.5*(target[1]+target[3])
                        bbox_reg[b, 2, sample_j, sample_i] = target[2] - target[0]
                        bbox_reg[b, 3, sample_j, sample_i] = target[3] - target[1]
                        bbox_reg[b, 4, sample_j, sample_i] = 0.5*(sample[0]+sample[2])
                        bbox_reg[b, 5, sample_j, sample_i] = 0.5*(sample[1]+sample[3])
                        bbox_reg[b, 6, sample_j, sample_i] = sample[2] - sample[0]
                        bbox_reg[b, 7, sample_j, sample_i] = sample[3] - sample[1]

        #normalize probabilities
        det_pr /= det_pr.sum(axis=1)[:,None,...]

        #normalize by number of samples
        nfactor = self.sample_num*self.sample_num
        det_pr /= nfactor
        if self.use_bbox_reg:
            bbox_valid /= nfactor

        #pack indices / values
        yt_value = det_pr.flatten()
        if self.use_bbox_reg:
            yt_value = numpy.concatenate((yt_value, bbox_valid.flatten(), bbox_reg.flatten()))

        return numpy.array([], dtype=numpy.int64), yt_value

    #
    def get_errors(self, yt_index, yt_value):

        #unpack indexs and values
        shapes = [self.det_shape]
        if self.use_bbox_reg:
            shapes += [(self.batch_size, self.sample_num, self.sample_num), (self.batch_size, 8, self.sample_num, self.sample_num)]
            
        v = common.ndarray_unpack(yt_value, shapes)
        det_pr = v[0]
        index = 1
        if self.use_bbox_reg:
            bbox_valid, bbox_reg = v[index:(index+2)]
            index += 2

        #Detection Cost:
        det_errors = -tensor.sum(det_pr*self.det_pr, axis=1) / math.log(self.det_shape[1])

        #Bounding Box Regression Cost:
        bbox_errors = None
        if self.use_bbox_reg:
            bbox_target = bbox_reg[:,0:4,...]
            bbox_sample = bbox_reg[:,4:8,...]
            
            #standard Fast R-CNN style cost
            tx = (bbox_target[:, 0, ...] - bbox_sample[:, 0, ...]) / bbox_sample[:, 2, ...]
            ty = (bbox_target[:, 1, ...] - bbox_sample[:, 1, ...]) / bbox_sample[:, 3, ...]
            tw = tensor.log(bbox_target[:, 2, ...] / bbox_sample[:, 2, ...])
            th = tensor.log(bbox_target[:, 3, ...] / bbox_sample[:, 3, ...])
            t = tensor.concatenate([tx[:,None, ...], ty[:,None, ...], tw[:,None, ...], th[:,None, ...]], axis=1)
            bbox_errors = tensor.sum(bbox_valid[:,None, ...] * theano_util.smooth_L1(t - self.bbox_reg), axis=1)

        return det_errors, bbox_errors

    #return negative log-likelihood training cost
    def cost(self, yt_index, yt_value):
        det_errors, bbox_errors = self.get_errors(yt_index, yt_value)
        cost = tensor.as_tensor_variable(0.0)
        if not det_errors is None:
            cost += theano.printing.Print('DET Cost:')(self.cost_factor * det_errors.sum() / self.batch_size)
        if not bbox_errors is None:
            cost += theano.printing.Print('BBOX Cost:')(self.bbox_factor * bbox_errors.sum() / self.batch_size)
        return cost

    #returns a list of most likely (class,bounding boxes)
    def get_detections(self, model, data_x, data_m, params):

        pr_threshold = params.get("prThreshold", 0.01)
        nms_threshold = params.get("nmsThreshold", 0.5)
        corner_threshold = params.get("cornerThreshold", self.sparse_layer.corner_threshold)
        corner_max = params.get("cornerMax", 1024)
        t = (pr_threshold, nms_threshold, corner_threshold, corner_max)
        logging.verbose("Using detection params - pr threshold: %f, nms threshold: %f, corner_threshold: %f, corner_max: %i"%t)

        first_detect = False
        if self.detect_func is None:

            #get all model outputs
            outputs=[]
            outputs.append(self.det_pr)
            if self.use_bbox_reg:
                outputs.append(self.bbox_reg)

            logging.info("Building detection function")
            self.detect_func = theano.function([model.input], outputs, givens=[(get_train(), tensor.cast(0, 'int8'))], on_unused_input='ignore')

            logging.verbose("Exporting graph...")
            with open("detect_graph.txt", "w") as f:
                theano.printing.debugprint(self.detect_func, file=f, print_type=True)
            
            first_detect = True

        #get sampling bounding boxs
        logging.verbose("Detecting sample bboxs (%.2f)"%corner_threshold)
        timer = common.Timer()
        sample_bboxs = self.sparse_layer.get_samples(data_x, train=False, store_shared=True)
        timer.mark()
        logging.verbose("Found sample bboxs: {}".format([len(bbox) for bbox in sample_bboxs]))

        #upload sampling bounding boxs
        bboxs = self.sparse_layer.set_samples(sample_bboxs)
        timer.mark()

        #classify sampling bounding boxs
        r = list(self.detect_func(data_x))

        #get outputs
        det_pr = r[0]
        r_index = 1
            
        if self.use_bbox_reg:
            bbox_reg = r[r_index]
            r_index += 1

            #update bbox array
            bboxs_cx = 0.5*(bboxs[:,:,:,0] + bboxs[:,:,:,2])
            bboxs_cy = 0.5*(bboxs[:,:,:,1] + bboxs[:,:,:,3])
            bboxs_w = bboxs[:,:,:,2] - bboxs[:,:,:,0]
            bboxs_h = bboxs[:,:,:,3] - bboxs[:,:,:,1]
            predict_cx = bbox_reg[:,0,:,:]*bboxs_w + bboxs_cx
            predict_cy = bbox_reg[:,1,:,:]*bboxs_h + bboxs_cy
            predict_w = numpy.exp(bbox_reg[:,2,:,:])*bboxs_w
            predict_h = numpy.exp(bbox_reg[:,3,:,:])*bboxs_h
            bboxs[:,:,:,0] = predict_cx - predict_w*0.5
            bboxs[:,:,:,1] = predict_cy - predict_h*0.5
            bboxs[:,:,:,2] = predict_cx + predict_w*0.5
            bboxs[:,:,:,3] = predict_cy + predict_h*0.5

        timer.mark()
        detlists = c_code.build_detections_nms(pr_threshold, nms_threshold, det_pr, bboxs, [len(s) for s in sample_bboxs])
        timer.mark()

        logging.verbose("Found detections:", [len(detlist) for detlist in detlists])
        logging.verbose("FPS=%.1f, Timing (ms) - get samples: %i, upload: %i, classify: %i, build+nms %i"%tuple([self.batch_size / timer.current()] + timer.deltas_ms()))

        if not first_detect:
            global detect_time, detect_num
            detect_time += timer.current()
            detect_num += self.batch_size
            logging.info("Average FPS=%.1f"%(detect_num / detect_time))

        #results format
        results=[]
        for i, detlist in enumerate(detlists):
            results.append({"detections":detlist, "meta":data_m[i]})

        return results


