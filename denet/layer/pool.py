#Max Pooling Layer
import math
import theano
import theano.tensor as tensor
import theano.sandbox.cuda.dnn as dnn

import denet.common.logging as logging
from denet.layer import AbstractLayer

class PoolLayer(AbstractLayer):
    type_name = "pool"

    def __init__(self, layers, size=(2, 2), stride=None, pad=(0,0), mode="max", ignore_border=True, json_param={}):
        super().__init__(layer_index=len(layers))

        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape

        self.size=json_param.get("size", size)
        self.pad=json_param.get("pad", pad)
        self.ignore_border=json_param.get("ignoreBorder", ignore_border)
        self.mode=json_param.get("mode", mode)
        self.stride=json_param.get("stride", stride)
        if self.stride is None:
            self.stride=self.size

        #output dim
        if self.ignore_border:
            h=int(math.floor((self.input_shape[2] + 2*self.pad[0] - self.size[0]) / self.stride[0])) + 1
            w=int(math.floor((self.input_shape[3] + 2*self.pad[1] - self.size[1]) / self.stride[1])) + 1
        else:
            h=int(math.ceil((self.input_shape[2] + 2*self.pad[0]) / self.stride[0]))
            w=int(math.ceil((self.input_shape[3] + 2*self.pad[1]) / self.stride[1]))

        #theano optimizer is sometimes failing to use cudnn pooling!
        use_cudnn = (dnn.dnn_available() and dnn.version() >= (4000,4000) and self.ignore_border)
        if use_cudnn:
            self.output = dnn.dnn_pool(self.input, ws=self.size, pad=self.pad, stride=self.stride, mode=self.mode)
        else:
            self.output = tensor.signal.pool.pool_2d(self.input, ds=self.size, padding=self.pad, ignore_border=self.ignore_border, st=self.stride, mode=self.mode)

        self.output_shape = (self.input_shape[0], self.input_shape[1], h, w)
        logging.verbose("Adding", self)

        # logging.verbose("Adding", self, "layer - input:", self.input_shape, "mode:", self.mode, "size:", self.size, "stride:", self.stride, "pad:", self.pad,
        #                 "ignore border:", self.ignore_border, "use cudnn:", use_cudnn)

    def parse_desc(layers, name, tags, params):
        if name != "P":
            return False

        size=(params.get(0), params.get(0))
        stride=(params.get(1, size[0]), params.get(1, size[0]))
        pad=(params.get(2,0), params.get(2,0))
        mode="average_inc_pad" if "A" in tags else "max"
        ignore_border = bool(not "B" in tags)
        layers.append(PoolLayer(layers, size, stride, pad, ignore_border=ignore_border, mode=mode))
        return True

    def export_json(self):
        json=super().export_json()
        json.update({"mode" : self.mode,
                     "size" : self.size,
                     "stride" : self.stride,
                     "pad" : self.pad,
                     "ignoreBorder" : self.ignore_border})
        return json


