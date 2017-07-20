#dummy layer for splitting the model into multiple evaluations
import numpy
import theano
import denet.common.logging as logging
from denet.layer import AbstractLayer

class SplitLayer(AbstractLayer):
    type_name = "split"
    
    def __init__(self, layers, json_param={}):
        super().__init__(layer_index=len(layers))

        self.enabled = json_param.get("enabled", True)
        self.has_split = self.enabled

        self.input = layers[-1].output
        self.output_shape = self.input_shape = layers[-1].output_shape
        if self.enabled:
            self.output = theano.shared(numpy.zeros(self.output_shape).astype(theano.config.floatX), str(self) + " - output")
        else:
            self.output = self.input
            
        logging.verbose("Adding", self, "layer - input:", self.input_shape, "enabled:", self.enabled)

    #on forward split phase store input in output
    def split_forward(self):
        return [(self.output, self.input)]

    #on backward split phase store gradient 
    def split_backward(self, cost, known_grads):
        return [(self.output, theano.tensor.grad(cost, self.output, known_grads=known_grads))]

    #known gradients after split_backward()
    def split_known_grads(self):
        return {self.input:self.output}

    def export_json(self):
        json = super().export_json()
        json.update({"enabled": self.enabled})
        return json
        
    def parse_desc(layers, name, tags, params):
        if name != "SPLIT":
            return False
        layers.append(SplitLayer(layers))
        return True
