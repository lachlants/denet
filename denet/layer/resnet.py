#Resnet Layer
import math
import theano
import theano.tensor as tensor

import denet.common.logging as logging
from denet.layer import AbstractLayer, InitialLayer, IdentityLayer
from denet.layer.batch_norm import BatchNormLayer
from denet.layer.batch_norm_relu import BatchNormReluLayer
from denet.layer.activation import ActivationLayer
from denet.layer.convolution import ConvLayer

class ResnetLayer(AbstractLayer):
    type_name = "resnet"

    #if use_original == True use the original resnet design, otherwise use pre-activation
    def __init__(self, layers, filter_shape=None, stride = (1,1), bottleneck=0, activation = "relu", version="original", json_param={}):
        super().__init__(layer_index=len(layers))

        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape

        #get parameters
        self.filter_shape = json_param.get("shape", filter_shape)
        self.stride = json_param.get("stride", stride)
        self.bottleneck = json_param.get("bottleneck", bottleneck)
        self.version = json_param.get("version", version)
        self.activation = json_param.get("activation", activation)
        self.bn_json_param = json_param.get("bnParam", {
            "enabled":json_param.get("enableBatchNorm", True),
            })
        
        #determing convolution shapes
        if self.bottleneck > 0:
            self.size = (self.filter_shape[2], self.filter_shape[3])
            shape0 = (self.bottleneck, self.filter_shape[1], 1, 1)
            shape1 = (self.bottleneck, self.bottleneck, self.filter_shape[2], self.filter_shape[3])
            shape2 = (self.filter_shape[0], self.bottleneck, 1, 1)
        else:
            self.size = (self.filter_shape[2]*2 - 1, self.filter_shape[3]*2 - 1)
            shape0 = self.filter_shape
            shape1 = (self.filter_shape[0], self.filter_shape[0], self.filter_shape[2], self.filter_shape[3])
            shape2 = None

        logging.verbose("Adding", self)

        # logging.verbose("Adding", self, "layer - input:", self.input_shape, "shape:", self.filter_shape, "stride:", self.stride, "bottleneck:", self.bottleneck,
        #                 "activation:", self.activation, "version:", self.version, "bn param:", self.bn_json_param)

        logging.verbose("--------RESNET BLOCK----------")

        self.layers = [InitialLayer(self.input, self.input_shape)]
        if "pre-activation" in self.version:
            if "bnrelu" in self.version and self.activation == "relu":
                self.layers.append(BatchNormReluLayer(self.layers, json_param = self.bn_json_param))
            else:
                self.layers.append(BatchNormLayer(self.layers, json_param = self.bn_json_param))
                self.layers.append(ActivationLayer(self.layers, self.activation))

        self.layers.append(ConvLayer(self.layers, filter_shape = shape0, filter_stride=self.stride, border_mode="half", use_bias=False))

        if "bnrelu" in self.version and self.activation == "relu":
            self.layers.append(BatchNormReluLayer(self.layers, json_param = self.bn_json_param))
        else:
            self.layers.append(BatchNormLayer(self.layers, json_param = self.bn_json_param))
            self.layers.append(ActivationLayer(self.layers, self.activation))

        self.layers.append(ConvLayer(self.layers, filter_shape = shape1, border_mode="half", use_bias=False))

        #for bottleneck design add additional conv
        if self.bottleneck > 0:
            if "bnrelu" in self.version and self.activation == "relu":
                self.layers.append(BatchNormReluLayer(self.layers, json_param = self.bn_json_param))
            else:
                self.layers.append(BatchNormLayer(self.layers, json_param = self.bn_json_param))
                self.layers.append(ActivationLayer(self.layers, self.activation))
            self.layers.append(ConvLayer(self.layers, filter_shape = shape2, border_mode="half", use_bias=False))

        if not "pre-activation" in self.version:
            self.layers.append(BatchNormLayer(self.layers, json_param = self.bn_json_param))

        y = self.layers[-1].output
        y_shape = self.layers[-1].output_shape

        #project input shape to output shape dimensions
        if self.input_shape != y_shape:

            logging.verbose("---------SHORTCUT----------")

            #handle resnet models with batchnorm in shortcut route
            if "pre-activation" in self.version:
                input_layers = self.layers[0:2]
            else:
                input_layers = [InitialLayer(self.input, self.input_shape)]

            self.layers.append(ConvLayer(input_layers, filter_shape=(y_shape[1], self.input_shape[1], 1, 1), filter_stride=self.stride, use_bias=False, border_mode="half"))

            #original model has batch norm after shortcut
            if "original" in self.version:
                self.layers.append(BatchNormLayer(self.layers, json_param = self.bn_json_param))

            x = self.layers[-1].output
        else:
            x = self.input

        logging.verbose("------------------------------")

        #add residual
        self.output_shape = y_shape
        if "pre-activation" in self.version:
            self.output = x + y
        else:
            self.output = ActivationLayer.apply(x + y, self.activation)

    def parse_desc(layers, name, tags, params):
        if name == "RSN":
            version = "original" if "O" in tags else "pre-activation"
            filter_shape=(params.get(0), layers[-1].output_shape[1], params.get(1), params.get(1))
            filter_stride=(params.get(2,1), params.get(2,1))
            bottleneck=params.get(3,0)
            layers.append(ResnetLayer(layers, filter_shape, filter_stride, bottleneck, params["activation"], version))
            return True

        elif name == "nRSN":
            version = "original" if "O" in tags else "pre-activation"
            bottleneck = params.get(4,0)
            for i in range(params.get(0)):
                filter_shape = (params.get(1), layers[-1].output_shape[1], params.get(2), params.get(2))
                filter_stride = (params.get(3,1), params.get(3,1)) if i == 0 else (1,1)
                layers.append(ResnetLayer(layers, filter_shape, filter_stride, bottleneck, params["activation"], version))
            return True

        return False

    def updates(self, cost):
        return sum([layer.updates(cost) for layer in self.layers], [])

    def weights(self):
        return sum([layer.weights() for layer in self.layers], [])

    def biases(self):
        return sum([layer.biases() for layer in self.layers], [])

    def import_json(self, json_param):

        n=0
        for json_layer in json_param["layers"]:

            #ignore identity layers introduced in old versions
            if json_layer["type"] == "identity":
                continue
            
            assert json_layer["type"] == self.layers[n].type_name, (json_layer["type"], self.layers[n].type_name)
            self.layers[n].import_json(json_layer)
            n+=1

    def export_json(self):
        json=super().export_json()
        json.update({"shape":self.filter_shape, "stride":self.stride, "bottleneck":self.bottleneck, "bnParam":self.bn_json_param,
                     "activation":self.activation, "version":self.version})

        json_layers=[]
        for layer in self.layers:
            json_layers.append(layer.export_json())

        json.update({"layers" : json_layers})
        return json


