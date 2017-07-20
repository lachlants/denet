#Convolution Layer
import math
import numpy
import theano
import theano.tensor as tensor
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs

import denet.common.logging as logging
from denet.layer import AbstractLayer

class DeconvLayer(AbstractLayer):
    type_name = "deconv"

    def __init__(self, layers, filter_shape=None, filter_stride=(1,1), use_bias=True, border_mode="valid", wb="he-backward", json_param={}):
        super().__init__(layer_index=len(layers))

        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape

        #get parameters
        self.border_mode = json_param.get("border", border_mode)
        self.filter_shape = tuple(json_param.get("shape", filter_shape))
        self.stride = tuple(json_param.get("stride", filter_stride))
        self.use_bias = json_param.get("useBias", use_bias)
        self.size = (self.filter_shape[2], self.filter_shape[3])

        #use initialization
        if type(wb) is float:
            self.w_bound = float(wb)
        elif "he-forward" in wb:
            self.w_bound = math.sqrt(2.0 / (self.filter_shape[2]*self.filter_shape[3]*self.filter_shape[1]))
        elif "he-backward" in wb:
            self.w_bound = math.sqrt(2.0 / (self.filter_shape[2]*self.filter_shape[3]*self.filter_shape[0]))
        elif "xavier-forward" in wb:
            self.w_bound = math.sqrt(1.0 / (self.filter_shape[2]*self.filter_shape[3]*self.filter_shape[1]))
        elif "xavier-backward" in wb:
            self.w_bound = math.sqrt(1.0 / (self.filter_shape[2]*self.filter_shape[3]*self.filter_shape[0]))

        #initialize weights
        if self.w_bound > 0:
            if "uniform" in wb:
                w = numpy.random.uniform(-self.w_bound, self.w_bound, size=self.filter_shape)
            else:
                w = numpy.random.normal(0.0, self.w_bound, size=self.filter_shape)
        else:
            w = numpy.zeros(shape=self.filter_shape)
        self.omega = theano.shared(numpy.asarray(w, dtype=theano.config.floatX), name="deconv omega")

        #initialize bias
        if self.use_bias:
            self.beta = theano.shared(value=numpy.zeros((self.filter_shape[0],), dtype=theano.config.floatX), name="deconv beta")

        #calculate output shape
        if self.border_mode == "half":
            fh = self.filter_shape[2] // 2
            fw = self.filter_shape[3] // 2
            h = self.input_shape[2]*self.stride[0] - 2*fh + self.filter_shape[2] - 1
            w = self.input_shape[3]*self.stride[1] - 2*fw + self.filter_shape[3] - 1
        else:
            raise Exception("Unknown border mode: " + str(self.border_mode))
            
        self.output_shape = (self.input_shape[0], self.filter_shape[0], h, w)
        self.output = conv2d_grad_wrt_inputs(self.input, self.omega.dimshuffle((1,0,2,3)), self.output_shape, 
                                             (self.filter_shape[1], self.filter_shape[0], self.filter_shape[2], self.filter_shape[3]), 
                                             self.border_mode, self.stride)

        if self.use_bias:
            self.output += self.beta[None,:,None,None]

        logging.verbose("Adding", self)

        # logging.verbose("Adding", self, "layer - input:", self.input_shape, "filter:", self.filter_shape, "stride:", self.stride, "use bias:", self.use_bias,
        #                 "wb: %.3f (%s)"%(self.w_bound, wb), "border_mode:", self.border_mode)

    def parse_desc(layers, name, tags, params):
        if name != "DC":
            return False

        use_bias = bool(not "B" in tags)
        if bool("X" in tags):
            filter_shape = (params.get(0), layers[-1].output_shape[1], params.get(1), params.get(2))
            filter_stride = (params.get(3,1), params.get(4,1))
        else:
            filter_shape=(params.get(0), layers[-1].output_shape[1], params.get(1,1), params.get(1,1))
            filter_stride=(params.get(2,1), params.get(2,1))

        layers.append(DeconvLayer(layers, filter_shape, filter_stride, use_bias, params["borderMode"], params["wb"]))
        return True

    def parameters(self):
        return [self.omega, self.beta] if self.use_bias else [self.omega]

    def weights(self):
        return [self.omega]

    def biases(self):
        return [self.beta] if self.use_bias else []

    def import_json(self, json_param):
        super().import_json(json_param)
        if self.use_bias:
            self.beta.set_value(numpy.asarray(json_param["bias"], dtype=theano.config.floatX), borrow=True)
        self.omega.set_value(numpy.asarray(json_param["weight"], dtype=theano.config.floatX), borrow=True)

    def export_json(self):
        json=super().export_json()
        json.update({"shape" : self.filter_shape,
                     "stride" : self.stride,
                     "border" : self.border_mode,
                     "useBias" : self.use_bias,
                     "bias" : self.beta.get_value() if self.use_bias else None,
                     "weight" : self.omega.get_value()})

        return json
