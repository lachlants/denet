import math
import theano
import theano.tensor as tensor

import denet.common.logging as logging
from denet.layer import AbstractLayer

class ActivationLayer(AbstractLayer):
    type_name = "activation"

    def __init__(self, layers, activation="relu", json_param={}):
        super().__init__(layer_index=len(layers))

        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape
        self.activation=json_param.get("activation", activation)

        self.output_shape = self.input_shape
        self.output = ActivationLayer.apply(self.input, self.activation)

        #apply activation
        logging.verbose("Adding", self)
        # logging.verbose("Adding", self, "layer - input:", self.input_shape, " activation:", self.activation)

    def apply(x, activation):
        if activation == "none":
            return x
        elif activation == "sigmoid":
            return tensor.nnet.sigmoid(x)
        elif activation == "relu-safe":
            return tensor.maximum(x, 0.0)
        elif activation == "relu":
            #Note: tensor.nnet.relu sometimes give small negative numbers!
            return tensor.nnet.relu(x)
        elif activation == "elu":
            return tensor.nnet.elu(x)
        elif activation == "tanh":
            return tensor.tanh(x)
        elif activation == "softmax":
            return tensor.nnet.softmax(x)
        elif activation == "softplus":
            return tensor.nnet.softplus(x)
        else:
            raise Exception("Unknown activation type:", activation)

    def parse_desc(layers, name, tags, params):
        if name != "A":
            return False

        layers.append(ActivationLayer(layers, params["activation"]))
        return True

    def export_json(self):
        json=super().export_json()
        json.update({"activation" : self.activation})
        return json
