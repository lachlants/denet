import math
import numpy
import theano
import theano.tensor as tensor

import denet.common.logging as logging
from denet.layer import AbstractLayer, get_train, get_rng

class DropoutLayer(AbstractLayer):
    type_name = "dropout"

    def __init__(self, layers, dropout_rate=1.0, json_param={}):
        super().__init__(layer_index=len(layers))

        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape

        #get parameters
        self.dropout_rate = json_param.get("dropoutRate", dropout_rate)

        scale = 1.0 / (1.0 - self.dropout_rate)
        mask = get_rng().binomial(n=1, p=1.0-self.dropout_rate, size=self.input_shape)
        mask = tensor.cast(mask, theano.config.floatX)
        self.output = theano.ifelse.ifelse(get_train(), self.input*mask*scale, self.input)
        self.output_shape = self.input_shape

        logging.verbose("Adding", self, "layer - input:", self.input_shape, "dropout: %.0f%%"%(100.0*self.dropout_rate))

    def parse_desc(layers, name, tags, params):
        if name != "D":
            return False
        dropout_rate = params.get(0, 0.5)
        layers.append(DropoutLayer(layers, dropout_rate))
        return True

    def export_json(self):
        json=super().export_json()
        json.update({"dropoutRate" : self.dropout_rate})
        return json
