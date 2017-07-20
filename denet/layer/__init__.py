import theano
import theano.tensor as tensor

#global variables to use to toggle training and rng, etc
from theano.sandbox.rng_mrg import MRG_RandomStreams
layer_train_rng = MRG_RandomStreams()
layer_train_enable = tensor.bscalar()
layer_train_epoch = tensor.iscalar()
layer_train_it = theano.shared(0)
def get_rng():
    global layer_train_rng
    return layer_train_rng

def set_rng_seed(v):
    global layer_train_rng
    layer_train_rng.seed(v)

def get_train():
    global layer_train_enable
    return layer_train_enable

def get_epoch():
    global layer_train_epoch
    return layer_train_epoch

def get_iteration():
    global layer_train_it
    return layer_train_it

#import a list of json_layer
def import_json(json_layers, x, x_shape, layer_range = None):

    if layer_range is None:
        layer_start = 0
        layer_end = len(json_layers)
    elif type(layer_range) is tuple:
        layer_start = layer_range[0]
        layer_end = min(len(json_layers), layer_range[1])
    elif type(layer_range) is int:
        layer_start = 0
        layer_end = min(len(json_layers), layer_range)
    else:
        raise Exception("Unknown layer range format:", layer_range)

    #delayed import to fix import cycle
    from denet.layer.layer_types import layer_types
    layers = [InitialLayer(x, x_shape)]
    for layer_json in json_layers[layer_start:layer_end]:

        layer=None
        for layer_type in layer_types:
            if layer_json["type"] == layer_type.type_name:
                layer = layer_type(layers, json_param = layer_json)
                break

        assert layer != None, "ERROR Unknown layer type: " + layer_json["type"]
        layer.import_json(layer_json)
        layers.append(layer)

    return layers


#abstract model layer
class AbstractLayer(object):

    def __init__(self, layer_index, has_split=False):
        self.output = self.input = None
        self.output_shape = self.input_shape = None
        self.has_split = has_split
        self.layers=[]
        self.layer_index = layer_index

    def __str__(self):

        param={"int":[], "str":[], "float":[], "bool":[], "tuple":[]}
        for k,v in self.__dict__.items():

            if k == "has_split" or k == "layer_index" or k == "output_shape": 
                continue

            if type(v) is int:
                param["int"].append(k + ": %i"%v)
            elif type(v) is str:
                param["str"].append(k + ": " + v)
            elif type(v) is float:
                param["float"].append(k + ": %.3f"%v)
            elif type(v) is bool:
                param["bool"].append(k + ": %s"%v)
            elif type(v) is tuple:
                param["tuple"].append(k + ": " + str(v))
        
        def get_param_str(param_type):
            if len(param[param_type]) > 0:
                param[param_type].sort()
                return " " + " ".join(param[param_type]) 
            else:
                return ""

        param_str = get_param_str("tuple") + get_param_str("str") + get_param_str("int") + get_param_str("float") + get_param_str("bool")
        return "%i:"%self.layer_index + self.type_name + " - " + param_str

    def weights(self):
        return sum([x.weights() for x in self.layers], []) 

    def biases(self):
        return sum([x.biases() for x in self.layers], [])

    #variable updates for split forward pass
    def split_forward(self):
        return []

    #variable updates for split backward pass
    def split_backward(self, cost, known_grads):
        return []

    #known grads after split_backward pass
    def split_known_grads(self):
        return {}

    #shared values to update
    def updates(self, cost):
        return sum([x.updates(cost) for x in self.layers], [])

    #all layer parameters
    def params(self):
        return self.weights() + self.biases()

    #additional per layer cost function (y = labels)
    def cost(self, yt_index, yt_value):
        return None

    #remap targets
    def get_target(self, model, samples, metas):
        return None

    #load / save
    def export_json(self):
        return {"type": type(self).type_name, "layers": [layer.export_json() for layer in self.layers]}

    def import_json(self, json_param):
        if "layers" in json_param:
            for i,json_layer in enumerate(json_param["layers"]):
                self.layers[i].import_json(json_layer)

#pass throught layer
class InitialLayer(AbstractLayer):
    type_name = "initial"
    def __init__(self, x, x_shape, json_param={}):
        super().__init__(layer_index=0)
        self.output = self.input = x
        self.output_shape = self.input_shape = x_shape
        # print("Adding initial layer - input: ", self.input_shape)

class IdentityLayer(AbstractLayer):
    type_name = "identity"
    def __init__(self, layers, json_param={}):
        super().__init__(layer_index=len(layers))
        self.output = self.input = layers[-1].output
        self.output_shape = self.input_shape = layers[-1].output_shape
        # print("Adding identity layer - input: ", self.input_shape)

    def parse_desc(layers, name, tags, params):
        return False
















