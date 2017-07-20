import numpy
import theano
import theano.tensor as tensor
import denet.common.logging as logging
from denet.layer import AbstractLayer, InitialLayer
from denet.layer.convolution import ConvLayer

#sum input from two different layers - optionally split model at this layer
class SkipSrcLayer(AbstractLayer):
    type_name = "skip-src"
    def __init__(self, layers, skip_index=0, split=False, json_param={}):
        super().__init__(layer_index=len(layers))
        self.skip_index = json_param.get("index", skip_index)
        self.has_split = json_param.get("split", split)
        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape

        self.output_shape = self.input_shape
        if self.has_split:
            self.output = theano.shared(numpy.zeros(self.output_shape).astype(theano.config.floatX), str(self) + " - output")
            self.skip = theano.shared(numpy.zeros(self.output_shape).astype(theano.config.floatX), str(self) + " - skip")
        else:
            self.skip = self.output = self.input

        logging.verbose("Adding", self, "split:", self.has_split)

    def export_json(self):
        j = super().export_json()
        j.update({"index" : self.skip_index, "split":self.has_split})
        return j

    #on forward split phase store input in output
    def split_forward(self):
        if self.has_split: 
            return [(self.output, self.input), (self.skip, self.input)]
        else:
            return []

    #on backward split phase 
    def split_backward(self, cost, known_grads):
        if self.has_split: 
            return [(self.output, theano.tensor.grad(cost, self.output, known_grads=known_grads))]
        else:
            return []
        
    def split_known_grads(self):
        if self.has_split: 
            return {self.input:(self.output + self.skip)}
        else:
            return {}

    def parse_desc(layers, name, tags, params):
        if name != "SKIPSRC":
            return False

        layers.append(SkipSrcLayer(layers, params.get(0, 0), "X" in tags))
        return True

class SkipLayer(AbstractLayer):
    type_name = "skip"
    def __init__(self, layers, skip_index=0, combine_mode="proj-add", json_param={}):
        super().__init__(layer_index=len(layers))

        self.combine_mode = json_param.get("combineMode", combine_mode)
        self.skip_index = json_param.get("index", skip_index)
        self.skip_layer=None
        for layer in layers:
            if layer.type_name == "skip-src" and layer.skip_index == self.skip_index:
                self.skip_layer=layer
                break
        assert self.skip_layer != None

        self.x = layers[-1].output
        self.x_shape = layers[-1].output_shape
        self.y = self.skip_layer.skip
        self.y_shape = self.skip_layer.output_shape

        if self.combine_mode == "proj-add":

            self.output_shape = self.x_shape
            if self.y_shape[1] != self.x_shape[1]:
                self.layers = [InitialLayer(self.y, self.y_shape)]
                self.layers.append(ConvLayer(self.layers, filter_shape = (self.x_shape[1], self.y_shape[1], 1, 1)))
                self.output = self.x + self.layers[-1].output
            else:
                self.output = self.x + self.y

        elif self.combine_mode == "concat":

            self.output_shape = (self.x_shape[0], self.x_shape[1] + self.y_shape[1], self.x_shape[2], self.x_shape[3])
            self.output = tensor.concatenate([self.x, self.y], axis=1)

        else:
            raise Exception("Unknown combine mode: %s"%self.combine_mode)

        logging.verbose("Adding", self, "layer - x:", self.x_shape, "y:", self.y_shape, "skip index:", self.skip_index, "combine mode:", self.combine_mode)

    #on split backward phase store skip layer grad
    def split_backward(self, cost, known_grads):
        if self.skip_layer.has_split: 
            return [(self.skip_layer.skip, theano.tensor.grad(cost, self.skip_layer.skip, known_grads=known_grads))]
        else:
            return []

    def export_json(self):
        j = super().export_json()
        j.update({"index" : self.skip_index, "combineMode":self.combine_mode})
        return j

    def parse_desc(layers, name, tags, params):
        if name != "SKIP":
            return False

        layers.append(SkipLayer(layers, params.get(0, 0)))
        return True

