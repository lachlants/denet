#Random Crop / Mirror Layer
import math
import numpy
import theano
import theano.tensor as tensor

import denet.common.logging as logging
from denet.layer import AbstractLayer, get_train, get_rng

class CropMirrorLayer(AbstractLayer):
    type_name = "crop-mirror"

    def __init__(self, layers, crop_size=None, mirror_pr = 0.0, flip_pr=0.0, json_param={}):
        super().__init__(layer_index=len(layers))

        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape

        self.crop_size = json_param.get("crop", crop_size)
        self.mirror_pr = json_param.get("mirror", mirror_pr)
        self.flip_pr = json_param.get("flip", flip_pr)
        self.output_shape = (self.input_shape[0], self.input_shape[1], self.crop_size[0], self.crop_size[1])
        self.output = []

        zero = tensor.zeros((self.input_shape[0],), dtype=numpy.int8)
        index_b = tensor.arange(self.input_shape[0])
        index_c = tensor.arange(self.input_shape[1])
        index_x = tensor.arange(self.crop_size[0])[None,:]
        index_y = tensor.arange(self.crop_size[1])[None,:]

        #randomly mirror (y-axis) data
        if self.mirror_pr > 0.0:
            mirror = tensor.gt(get_rng().uniform(size=(self.input_shape[0],)), 1.0 - self.mirror_pr)
            mirror = tensor.switch(get_train(), mirror, zero)
            index_y = tensor.switch(mirror[:,None], -index_y + self.crop_size[1] - 1, index_y)

        #randomly flip (x-axis) data
        if self.flip_pr > 0.0:
            flip = tensor.gt(get_rng().uniform(size=(self.input_shape[0],)), 1.0 - self.flip_pr)
            flip = tensor.switch(get_train(), flip, zero)
            index_x = tensor.switch(flip[:,None], -index_x + self.crop_size[0] - 1, index_x)

        #randomly offset crop
        dx = self.input_shape[2] - self.crop_size[0]
        dy = self.input_shape[3] - self.crop_size[1]
        if self.crop_size[0] != self.input_shape[2] or self.crop_size[1] != self.input_shape[3]:
            center_x = theano.shared(numpy.full(shape=(self.input_shape[0],), fill_value=dx // 2, dtype=numpy.int32), borrow=False)
            center_y = theano.shared(numpy.full(shape=(self.input_shape[0],), fill_value=dy // 2, dtype=numpy.int32), borrow=False)
            offset_x = get_rng().random_integers(size=(self.input_shape[0],), low=0, high=dx)
            offset_y = get_rng().random_integers(size=(self.input_shape[0],), low=0, high=dy)
            index_x += tensor.switch(get_train(), offset_x, center_x)[:,None]
            index_y += tensor.switch(get_train(), offset_y, center_y)[:,None]

        #perform advanced indexing
        self.output = self.input[index_b[:,None,None,None], index_c[None,:,None,None], index_x[:,None,:,None], index_y[:,None,None,:]]

        logging.verbose("Adding", self)

        # logging.verbose("Adding", self, "layer - input:", self.input_shape, "crop size:", self.crop_size, "mirror pr:", self.mirror_pr, 
        #                 "flip pr:", self.flip_pr, "test position:", (dx//2, dy//2))

    def parse_desc(layers, name, tags, params):
        if name != "CM":
            return False

        crop_size = (params.get(0), params.get(0))
        mirror_pr = params.get(1,0.0)
        flip_pr = params.get(2,0.0)
        layers.append(CropMirrorLayer(layers, crop_size, mirror_pr, flip_pr))
        return True

    def export_json(self):
        json=super().export_json()
        json.update({"crop" : self.crop_size, "mirror": self.mirror_pr, "flip": self.flip_pr})
        return json

#testing
if __name__ == '__main__':
    from theano.sandbox.rng_mrg import MRG_RandomStreams
    import sys

    seed= int(sys.argv[1])
    rng = MRG_RandomStreams(seed=seed, use_cuda=True)

    batch_size = 32
    width = 4
    height = 4
    crop_width = 2
    crop_height = 2

    input_shape = (batch_size, 3, width, height)
    input_x = numpy.asarray(numpy.random.uniform(size=input_shape, low=0, high=1), dtype=theano.config.floatX)
    input = theano.shared(input_x, borrow=False)


    train_switch_val = theano.shared(1.0)
    train_switch = tensor.gt(train_switch_val, 0.5)

    layer = CropMirrorLayer(input, input_shape, train_switch, rng, (crop_width, crop_height), 0.0, 0.0)

    output_func = theano.function([], [input, layer.output[0]])
    x,y = output_func()

    theano.printing.debugprint(output_func.maker.fgraph.outputs[1])

    for i in range(batch_size):
        print("input[%i,0,:,:]\n"%i, input_x[i,0,:,:])
        print("output[%i,0,:,:]\n"%i, y[i,0,:,:])
