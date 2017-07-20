#Max Pooling Layer
import math
import theano
import theano.tensor as tensor

import denet.common.logging as logging
from denet.layer import AbstractLayer, get_train
from denet.layer.pool_inv_op import PoolInvOp

class PoolInvLayer(AbstractLayer):
    type_name = "pool-inv"

    def __init__(self, layers, size=(2, 2), json_param={}):
        super().__init__(layer_index=len(layers))
        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape

        self.size=json_param.get("size", size)

        #output dim
        self.output_shape = (self.input_shape[0], self.input_shape[1], self.size[1]*self.input_shape[2], self.size[0]*self.input_shape[3])
        self.use_optimized = theano.sandbox.cuda.cuda_enabled
        if self.use_optimized:
            self.output = PoolInvOp(self.size)(self.input)
        else:
            self.output = tensor.repeat(tensor.repeat(self.input, self.size[1], axis=2), self.size[0], axis=3)
            
        logging.verbose("Adding", self)

    def parse_desc(layers, name, tags, params):
        if name != "PI":
            return False

        size=(params.get(0), params.get(0))
        layers.append(PoolInvLayer(layers, size))
        return True

    def export_json(self):
        json=super().export_json()
        json.update({"size" : self.size})
        return json

if __name__ == '__main__':
        
    logging.init()
    from denet.layer import InitialLayer
    import time
    import numpy

    batch_size=4
    feature_num=64
    height=4
    width=4
    grid_size=(2,2)

    numpy.random.seed(1)
    x = numpy.random.uniform(-5, 5, (batch_size, feature_num, height, width)).astype(numpy.float32)
    
    fmap = tensor.tensor4()
    fmap_shape = x.shape
    pool_inv_new = PoolInvOp(grid_size)(fmap)
    pool_inv_old = tensor.repeat(tensor.repeat(fmap, grid_size[1], axis=2), grid_size[0], axis=3)

    print("Building function")
    f_new = theano.function([fmap], pool_inv_new)
    f_old = theano.function([fmap], pool_inv_old)
    f_new_grad = theano.function([fmap], tensor.grad(pool_inv_new.sum(), fmap))
    f_old_grad = theano.function([fmap], tensor.grad(pool_inv_old.sum(), fmap))

    print("------- OLD -------")
    r_old = numpy.array(f_old(x))
    print(r_old.shape)
    print(r_old[:,0,:,:])

    print("------- NEW -------")
    r_new = numpy.array(f_new(x))
    print(r_new.shape)
    print(r_new[:,0,:,:])

    print("------- OLD Grad-------")
    r_old = numpy.array(f_old_grad(x))
    print(r_old.shape)
    print(r_old[:,0,:,:])

    print("------- NEW Grad-------")
    r_new = numpy.array(f_new_grad(x))
    print(r_new.shape)
    print(r_new[:,0,:,:])
