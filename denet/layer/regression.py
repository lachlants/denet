import numpy
import theano
import theano.tensor as tensor

import denet.common.logging as logging
from denet.layer import AbstractLayer
from denet.layer.convolution import ConvLayer

#Generates probabilities and cost functions from feeatures
class RegressionLayer(AbstractLayer):

    type_name = "regression"

    def __init__(self, layers, use_center=True, valid=[], json_param={}):
        super().__init__(layer_index=len(layers))

        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape

        #used for determining multiview map
        self.multiview_layers = layers

        if use_center:
            yc = self.input_shape[-2] // 2
            xc = self.input_shape[-1] // 2
            valid = [(0,yc,xc)]

        self.valid = json_param.get("valid", valid)

        #could be done faster with advanced indexing
        if len(self.valid) > 0:
            x_shape = (self.input_shape[0], self.input_shape[1], len(self.valid))
            x = tensor.zeros(x_shape)
            for i, offset in enumerate(self.valid):
                x = tensor.set_subtensor(x[:,:,i], self.input[:,:,offset[1],offset[2]])
        else:
            x = self.input
            x_shape = self.input_shape

        self.log_likelihood = x
        self.log_pr = self.log_softmax(x, axis=1)
        self.log_pr_shape = x_shape

        #return probabilities as output
        self.output_shape = (self.log_pr_shape[0], self.log_pr_shape[1])
        if len(self.log_pr_shape) > 2:
            self.output = tensor.exp(self.log_pr).mean(axis=range(2, len(self.log_pr_shape)))
        else:
            self.output = tensor.exp(self.log_pr)

        logging.verbose("Adding", self, "layer - input:", self.input_shape, "valid:", self.valid)

    def parse_desc(layers, name, tags, params):
        if name != "R":
            return False

        use_bias = bool("B" in tags)
        use_center = bool("C" in tags)
        wb = "uniform,xavier-forward" if "T" in tags else 0.0
        filter_shape=(params["classNum"], layers[-1].output_shape[1], params.get(0,layers[-1].output_shape[2]), params.get(0,layers[-1].output_shape[3]) )
        layers.append(ConvLayer(layers, filter_shape, (1,1), use_bias, "valid", params["wb"]))
        layers.append(RegressionLayer(layers, use_center))
        return True

    #numerically stable log-softmax
    def log_softmax(self, x, axis):
        xdev = x - x.max(axis=axis, keepdims=True)
        return xdev - tensor.log(tensor.sum(tensor.exp(xdev), axis=axis, keepdims=True))

    def export_json(self):
        json = super().export_json()
        json.update({"valid" : self.valid})
        return json

    #convert meta data for samples into (yt_index, yt_value) for training cost
    def get_target(self, model, samples, metas):

        yt_index=[]
        if len(self.output_shape) == 2:
            for b in range(len(metas)):
                yt_index.append(numpy.ravel_multi_index((b, metas[b]["image_class"]), self.output_shape))

        elif len(self.valid) > 0:
            for b in range(len(metas)):
                for v in range(len(self.valid)):
                    yt_index.append(numpy.ravel_multi_index((b, metas[b]["image_class"], v), self.output_shape))
        else:
            for b in range(len(metas)):
                cls = metas[b]["image_class"]
                for y in range(self.output_shape[2]):
                    for x in range(self.output_shape[3]):
                        yt_index.append(numpy.ravel_multi_index((b, metas[b]["image_class"], y, x), self.output_shape))

        return numpy.array(yt_index, dtype=numpy.int64), numpy.array([], dtype=theano.config.floatX)

    #return negative log-likelihood training cost (scalar)
    def cost(self, yt_index, yt_value):
        return -(self.log_pr.flatten()[yt_index]).mean()
