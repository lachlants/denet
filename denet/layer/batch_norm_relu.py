import os
import numpy
import theano
import theano.tensor as tensor

from theano.gof import COp
from theano.sandbox.cuda import dnn_version
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.dnn import GpuDnnBatchNorm, GpuDnnBatchNormGrad, dnn_batch_normalization_train, dnn_batch_normalization_test

import denet.common.logging as logging
from denet.layer import AbstractLayer, get_train, get_epoch

#performs batch normalization followed by RELU to reduce memory requirements
class BatchNormReluOp(GpuDnnBatchNorm):

    def __init__(self, mode='per-activation', epsilon=1e-4):

        if dnn_version() < (5000, 5000):
            raise RuntimeError("cuDNN Batch Normalization requires cuDNN v5")

        assert (mode in ('per-activation', 'spatial'))
        assert (epsilon >= 1e-5)

        fname = os.path.join(os.path.dirname(theano.sandbox.cuda.dnn.__file__), "dnn_base.c")
        COp.__init__(self, fname)
        self.mode = mode
        self.epsilon = epsilon

    #relu kernel
    def c_support_code(self):
        result = super(BatchNormReluOp, self).c_support_code()
        return result + """
        static __global__ void k_relu(float* x, size_t x_num){
            size_t i = blockIdx.x*blockDim.x + threadIdx.x;
            if (i < x_num){    
                x[i] = (x[i] + fabsf(x[i]))/2;
            }
        };"""

    def c_code(self, node, name, inputs, outputs, sub):
        result = super(BatchNormReluOp, self).c_code(node, name, inputs, outputs, sub)
        return result + """{
        size_t num = CudaNdarray_SIZE(%s);
        size_t threads_per_block = 1024;
        size_t grid_num = std::ceil((double)num / threads_per_block);
        k_relu<<<grid_num, threads_per_block>>>(CudaNdarray_DEV_DATA(%s), num);
        };"""%(outputs[0], outputs[0])

    def grad(self, inputs, grads):
        x, scale, bias = inputs
        xn, x_mean, x_invstd = self(x, scale, bias)
        dy_relu = tensor.switch(xn > 0.0, grads[0], 0.0)
        return GpuDnnBatchNormGrad(self.mode, self.epsilon)(x, dy_relu, scale, x_mean, x_invstd)

    def c_code_cache_version(self):
        return (2, 8)

def dnn_bnrelu_train(inputs, gamma, beta, mode='per-activation', epsilon=1e-4):
    ndim = inputs.ndim
    if ndim > 5:
        raise ValueError("dnn_batch_normalization_train currently supports "
                         "up to 5-dimensional tensors only, got %d" % ndim)
    if gamma.ndim != ndim or beta.ndim != ndim:
        raise ValueError("gamma and beta must be of the same dimensionality "
                         "as inputs; got %d and %d instead of %d" %
                         (gamma.ndim, beta.ndim, ndim))
    if epsilon < 1e-5:
        raise ValueError("epsilon must be at least 1e-5, got %f" % epsilon)

    if ndim < 4:
        inputs = theano.tensor.shape_padright(inputs, 4 - ndim)
        gamma = theano.tensor.shape_padright(gamma, 4 - ndim)
        beta = theano.tensor.shape_padright(beta, 4 - ndim)

    batchnorm_op = BatchNormReluOp(mode=mode, epsilon=epsilon)
    result = tuple(batchnorm_op(gpu_contiguous(inputs), gpu_contiguous(gamma), gpu_contiguous(beta)))
    if ndim < 4:
        result = tuple(theano.tensor.flatten(r, ndim) for r in result)
    return result

def dnn_bnrelu_test(inputs, gamma, beta, mean, var,  mode='per-activation', epsilon=1e-4):
    return tensor.nnet.relu(dnn_batch_normalization_test(inputs, gamma, beta, mean, var, mode, epsilon))

class BatchNormReluLayer(AbstractLayer):
    type_name = "batchnorm-relu"

    def __init__(self, layers, momentum=0.9, eps=1e-5, json_param={}):
        super().__init__(layer_index=len(layers))

        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape

        #get parameters
        self.enabled = json_param.get("enabled", True)
        self.momentum = json_param.get("momentum", momentum)
        self.eps = json_param.get("eps", eps)

        #initialize param
        param_shape = (self.input_shape[1],)
        self.omega = theano.shared(numpy.asarray(numpy.ones(param_shape), dtype=theano.config.floatX), name="bn omega")
        self.beta = theano.shared(numpy.asarray(numpy.zeros(param_shape), dtype=theano.config.floatX), name="bn beta")
        self.mean = theano.shared(numpy.asarray(numpy.zeros(param_shape), dtype=theano.config.floatX), name="bn mean")
        self.stdinv = theano.shared(numpy.asarray(numpy.ones(param_shape), dtype=theano.config.floatX), name="bn std inv")

        x_shape = self.input_shape
        x = self.input
        dim = ['x',0,'x','x']
        use_cudnn = theano.sandbox.cuda.dnn.dnn_available() and (theano.sandbox.cuda.dnn.version() >= (5000,5000))
        # use_cudnn = theano.gpuarray.dnn.dnn_available(None) and (theano.gpuarray.dnn.version() >= 5000)
        if use_cudnn:
            var = tensor.sqr(1.0 / self.stdinv)
            x_n_train, x_mean, x_stdinv = dnn_bnrelu_train(x, self.omega.dimshuffle(dim), self.beta.dimshuffle(dim), 'spatial', self.eps)
            x_n_test = dnn_bnrelu_test(x, self.omega.dimshuffle(dim), self.beta.dimshuffle(dim), self.mean.dimshuffle(dim), var.dimshuffle(dim), 'spatial', self.eps)
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

        self.local_updates = [(self.mean, self.momentum*self.mean + (1.0 - self.momentum)*x_mean.squeeze()),
                              (self.stdinv, self.momentum*self.stdinv + (1.0 - self.momentum)*x_stdinv.squeeze())]

        self.output_shape = self.input_shape
        self.output = tensor.switch(get_train(), tensor.as_tensor_variable(x_n_train), theano.gradient.disconnected_grad(x_n_test)) 

        logging.verbose("Adding", self)
        # logging.verbose("Adding", self, "layer - input: ", self.input_shape, "momentum:", self.momentum, "eps:", self.eps, "use cudnn:", use_cudnn)

    def parse_desc(layers, name, tags, params):
        if name != "BNA":
            return False
 
        layers.append(BatchNormReluLayer(layers, params.get(0, 0.9), params.get(1, 1e-5)))
        return True

    #make sure mean / std included in parameter list
    def params(self):
        return [self.omega, self.beta, self.mean, self.stdinv]

    def updates(self, cost):
        return self.local_updates

    def biases(self):
        return [self.omega, self.beta]

    def export_json(self):
        json = super().export_json()
        json.update({"momentum": self.momentum,
                     "eps": self.eps,
                     "mean" : self.mean.get_value(borrow=True),
                     "std" : self.stdinv.get_value(borrow=True),
                     "gamma" : self.omega.get_value(borrow=True),
                     "bias" : self.beta.get_value(borrow=True)})
        return json

    def import_json(self, json_param):
        self.omega.set_value(numpy.asarray(json_param["gamma"], dtype=theano.config.floatX), borrow=True)
        self.beta.set_value(numpy.asarray(json_param["bias"], dtype=theano.config.floatX), borrow=True)
        self.mean.set_value(numpy.asarray(json_param["mean"], dtype=theano.config.floatX), borrow=True)
        self.stdinv.set_value(numpy.asarray(json_param["std"], dtype=theano.config.floatX), borrow=True)


if __name__ == '__main__':
        
    shape=(1,4,4,4)
    f_g = theano.shared(numpy.ones(shape=(shape[1],), dtype='float32'))
    f_b = theano.shared(numpy.zeros(shape=(shape[1],), dtype='float32'))
    f_x = theano.tensor.tensor4()

    f0_y, f0_mean, f0_std = dnn_bnrelu_train(f_x, f_g[None,:,None,None], f_b[None,:,None,None], "spatial")
    f0_yg = theano.tensor.grad(f0_y.sum(), f_x)
    f0_yg = theano.printing.Print('R0:')(f0_yg)
    f0 = theano.function([f_x], [f0_y, f0_y.sum(), f0_yg])

    f1_xn, f1_mean, f1_std = dnn_batch_normalization_train(f_x, f_g[None,:,None,None], f_b[None,:,None,None], "spatial")
    f1_y = tensor.maximum(f1_xn, 0.0)
    f1_yg = theano.tensor.grad(f1_y.sum(), f_x)
    f1_yg = theano.printing.Print('R1:')(f1_yg)
    f1 = theano.function([f_x], [f1_y, f1_y.sum(), f1_yg])
    
    x = numpy.random.uniform(-5.0, 5.0, shape).astype(numpy.float32)
    y0, ys0, yy0 = f0(x)
    y1, ys1, yy1 = f1(x)

    print("X Mean:", x.mean(axis=(0,2,3)))
    print("X std:", x.std(axis=(0,2,3)))

    print("------X------")
    print(numpy.array(x))

    print("------Y0------")
    print(numpy.array(y0))
    print("Sum:", ys0)

    print("------Y1------")
    print(numpy.array(y1))
    print("Sum:", ys1)

    print("------YY0------")
    print(numpy.array(yy0))

    print("------YY1------")
    print(numpy.array(yy1))
    
    
