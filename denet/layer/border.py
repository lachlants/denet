import math
import numpy
import theano
import theano.tensor as tensor

import denet.common.logging as logging
from denet.layer import AbstractLayer

class BorderLayer(AbstractLayer):
    type_name = "border"

    def __init__(self, layers, border = 0, json_param={}):
        super().__init__(layer_index=len(layers))

        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape

        #border = (Left, Right, Top, Bottom)
        if type(border) is int:
            border = (border, border, border, border)
        elif len(border) == 1:
            border = (border[0], border[0], border[0], border[0])

        assert len(border) == 4
        self.border = json_param.get("border", border)

        self.output_shape = list(self.input_shape)
        self.output_shape[-1] += self.border[0]+self.border[1]
        self.output_shape[-2] += self.border[2]+self.border[3]
        self.output_shape = tuple(self.output_shape)

        self.output = tensor.zeros(self.output_shape)
        self.output = tensor.set_subtensor(self.output[:,:, self.border[2]:(self.input_shape[-2]+self.border[2]), self.border[0]:(self.input_shape[-1]+self.border[0])], self.input)

        logging.verbose("Adding", self)

    def parse_desc(layers, name, tags, params):
        if name != "B":
            return False
        layers.append(BorderLayer(layers, params.get(0,0)))
        return True

    def export_json(self):
        json=super().export_json()
        json.update({"border" : self.border})
        return json
