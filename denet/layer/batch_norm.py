import operator
import logging
import numpy
import theano
import theano.tensor as tensor
import theano.sandbox.cuda.dnn
# import theano.gpuarray.dnn

import denet.common.logging as logging
from denet.layer import AbstractLayer, get_train, get_epoch

class BatchNormLayer(AbstractLayer):
    type_name = "batchnorm"

    def __init__(self, layers, momentum=0.9, eps=1e-5, renorm_max_r=1.0, renorm_max_d=0.0, renorm_max_it=10, json_param={}):
        super().__init__(layer_index=len(layers))
 
        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape

        #get parameters
        self.enabled = json_param.get("enabled", True)
        self.momentum = json_param.get("momentum", momentum)

        self.renorm_max_r = json_param.get("renormMaxR", renorm_max_r)
        self.renorm_max_d = json_param.get("renormMaxD", renorm_max_d)
        self.renorm_max_it = json_param.get("renormMaxIt", renorm_max_it)
 
        self.eps = json_param.get("eps", eps)
        if self.enabled:

            #initialize param
            param_shape = (self.input_shape[1],)
            self.omega = theano.shared(numpy.asarray(numpy.ones(param_shape), dtype=theano.config.floatX), name="bn omega")
            self.beta = theano.shared(numpy.asarray(numpy.zeros(param_shape), dtype=theano.config.floatX), name="bn beta")
            self.mean = theano.shared(numpy.asarray(numpy.zeros(param_shape), dtype=theano.config.floatX), name="bn mean")
            self.stdinv = theano.shared(numpy.asarray(numpy.ones(param_shape), dtype=theano.config.floatX), name="bn std inv")

            #evaluate
            x_shape = self.input_shape
            x = self.input

            #directly call cudnn version until added to master
            dim = ['x',0,'x','x']
            use_cudnn = theano.sandbox.cuda.dnn.dnn_available() and (theano.sandbox.cuda.dnn.version() >= (5000,5000))
            # use_cudnn = theano.gpuarray.dnn.dnn_available(None) and (theano.gpuarray.dnn.version() >= 5000)
            if use_cudnn:
                from theano.sandbox.cuda.dnn import dnn_batch_normalization_train, dnn_batch_normalization_test
                # from theano.gpuarray.dnn import dnn_batch_normalization_train, dnn_batch_normalization_test
                var = tensor.sqr(1.0 / self.stdinv)
                x_n_train, x_mean, x_stdinv = dnn_batch_normalization_train(x, self.omega.dimshuffle(dim), self.beta.dimshuffle(dim), 'spatial', self.eps)
                x_n_test = dnn_batch_normalization_test(x, self.omega.dimshuffle(dim), self.beta.dimshuffle(dim), self.mean.dimshuffle(dim), var.dimshuffle(dim), 'spatial', self.eps)
                x_std = 1.0 / x_stdinv
            else:
                #WARNING: BROKEN!
                xt = x.dimshuffle((1,0,2,3)).flatten(2)
                x_mean = tensor.sum(xt, axis=1) / (self.input_shape[0]*self.input_shape[2]*self.input_shape[3])
                x_mean = tensor.cast(x_mean, "float32")
                x_std = tensor.sqrt(tensor.mean(x*x, axis=[0,2,3]) - x_mean*x_mean + self.eps)
                x_stdinv = 1.0 / x_std

                x_n_test = (x - self.mean.dimshuffle(dim)) * (self.omega * self.stdinv).dimshuffle(dim) + self.beta.dimshuffle(dim)
                x_n_train = (x - x_mean.dimshuffle(dim)) * (self.omega * x_stdinv).dimshuffle(dim) + self.beta.dimshuffle(dim)

            #override old value with renormalized version
            # if (self.renorm_max_r > 1.0) or (self.renorm_max_d > 0.0):
            #     r_alpha = math.log(self.renorm_max_r) / self.renorm_max_it
            #     d_alpha = math.log(self.renorm_max_d + 1) / self.renorm_max_it
            #     r_max = tensor.minimum(self.renorm_max_r, tensor.exp(get_epoch()*r_alpha))
            #     d_max = tensor.minimum(self.renorm_max_d, tensor.exp(get_epoch()*d_alpha) - 1)
            #     x_r = tensor.gradient.zero_grad(tensor.clip(x_std*self.stdinv, 1.0/r_max, r_max))
            #     x_d = tensor.gradient.zero_grad(tensor.clip((x_mean - self.mean) * self.stdinv, -d_max, d_max))
            #     x_n_train = (x - x_mean.dimshuffle(dim)) * (self.omega*x_stdinv*x_r).dimshuffle(dim) + (self.beta + self.omega*x_d).dimshuffle(dim)

            self.local_updates = [(self.mean, self.momentum*self.mean + (1.0 - self.momentum)*x_mean.squeeze()),
                                  (self.stdinv, self.momentum*self.stdinv + (1.0 - self.momentum)*x_stdinv.squeeze())]

            self.output_shape = self.input_shape
            self.output = tensor.switch(get_train(), tensor.as_tensor_variable(x_n_train), theano.gradient.disconnected_grad(x_n_test)) 
        else:
            self.output_shape = self.input_shape
            self.output = self.input

        logging.verbose("Adding", self)
        # logging.verbose("Adding", self, "layer - input: ", self.input_shape, "momentum:", self.momentum, "eps:", self.eps, 
        #                 "renorm:", (self.renorm_max_r, self.renorm_max_d, self.renorm_max_it), "use cudnn:", use_cudnn, 
        #                 "enabled:", self.enabled)

    def parse_desc(layers, name, tags, params):
        if name != "BN":
            return False
 
        renorm_max_r = params.get(2, 1) 
        renorm_max_d = params.get(3, 0) 
        renorm_max_it = params.get(4, 0) 
        layers.append(BatchNormLayer(layers, params.get(0, 0.9), params.get(1, 1e-5), renorm_max_r, renorm_max_d, renorm_max_it))
        return True

    #make sure mean / std included in parameter list
    def params(self):
        return [self.omega, self.beta, self.mean, self.stdinv] if self.enabled else []

    def updates(self, cost):
        return self.local_updates if self.enabled else []

    def biases(self):
        return [self.omega, self.beta] if self.enabled else []

    def export_json(self):
        json = super().export_json()
        json.update({"momentum": self.momentum,
                     "eps": self.eps,
                     "mean" : self.mean.get_value(borrow=True),
                     "std" : self.stdinv.get_value(borrow=True),
                     "gamma" : self.omega.get_value(borrow=True),
                     "bias" : self.beta.get_value(borrow=True),
                     "renormMaxR" : self.renorm_max_r,
                     "renormMaxD" : self.renorm_max_d,
                     "renormMaxIt" : self.renorm_max_it,
                     "enabled": self.enabled})
        return json

    def import_json(self, json_param):
        if self.enabled:
            self.omega.set_value(numpy.asarray(json_param["gamma"], dtype=theano.config.floatX), borrow=True)
            self.beta.set_value(numpy.asarray(json_param["bias"], dtype=theano.config.floatX), borrow=True)
            self.mean.set_value(numpy.asarray(json_param["mean"], dtype=theano.config.floatX), borrow=True)
            self.stdinv.set_value(numpy.asarray(json_param["std"], dtype=theano.config.floatX), borrow=True)
           
#TESTING
def test():
    from denet.layer import InitialLayer
    numpy.random.seed(1002)
    eps = 1e-4

    input = tensor.tensor4()
    input_shape = (64,128,32,32)
    bn = BatchNormLayer([InitialLayer(input, input_shape)])
    f = theano.function([input], bn.output, updates=bn.local_updates, givens=[(get_train(), tensor.cast(1, 'int8'))])
    x = numpy.random.uniform(0.0, 1.0, input_shape).astype(numpy.float32)
    y = f(x)
    x_mean = bn.mean.get_value()
    x_std = bn.std.get_value()
    
    import theano
    import numpy
    input = theano.tensor.tensor4()
    f = theano.function([input], input.mean())
    x = numpy.random.uniform(0.0, 1.0, (64,128,32,32)).astype(numpy.float32)
    print("Mean TEST = ", f(x))
    
    
    if abs(y.mean()) > eps or abs(y.std() - 1.0) > eps or abs(x_mean.mean() - x.mean()*0.1) > eps or abs(x_std.mean() - 1.24641) > eps:
        raise Exception("Batchnorm failed test! ", y.mean(), y.std(), x_mean.mean(), x_std.mean())

if __name__ == '__main__':
    import sys
    sys.exit(test())
