import theano
import theano.tensor as tensor
import numpy
import math
import time

import denet.common as common
import denet.common.logging as logging
import denet.common.theano_util as theano_util

from denet.layer import AbstractLayer, InitialLayer
from denet.layer.convolution import ConvLayer
from denet.layer.pool_inv import PoolInvLayer
from denet.layer.activation import ActivationLayer
from denet.layer.batch_norm import BatchNormLayer

class DeNetCornerLayer(AbstractLayer):

    type_name = "denet-corner"
    def __init__(self, layers, sample_feat=512, cost_factor=1, dropout=0.0, use_center=False, json_param={}):
        super().__init__(layer_index=len(layers))

        #pass though layer
        self.input = layers[-1].output
        self.input_shape = layers[-1].output_shape
        self.output = layers[-1].output
        self.output_shape = layers[-1].output_shape

        self.batch_size, self.features, self.height, self.width = self.input_shape

        #get param
        self.sample_feat = json_param.get("sampleFeat", sample_feat)
        self.cost_factor = json_param.get("costFactor", cost_factor)
        self.use_center = json_param.get("useCenter", use_center)
        self.dropout = json_param.get("dropout", dropout)

        self.corner_num = 5 if self.use_center else 4
        self.layers = [InitialLayer(self.input, self.input_shape)]
        self.layers.append(ConvLayer(self.layers, (self.corner_num + self.sample_feat, self.features, 1, 1), (1,1), True, False))

        #initialize corner_pr weights / biases
        omega = self.layers[-1].omega.get_value()
        omega[:self.corner_num,:,:,:] = 0.0
        self.layers[-1].omega.set_value(omega)
        beta = self.layers[-1].beta.get_value()
        beta[:self.corner_num] = 5.0
        self.layers[-1].beta.set_value(beta)

        #extract corner probabilities
        self.corner_shape = (self.batch_size, 2, self.corner_num, self.height, self.width)
        self.corner_lh = self.layers[-1].output[:,:self.corner_num,:,:]
        self.corner_lh = tensor.concatenate([self.corner_lh[:,None,...], -self.corner_lh[:,None,...]], axis=1)
        self.corner_pr = theano_util.log_softmax(self.corner_lh, axis=[1])

        #extract sample
        self.sample_shape = (self.batch_size, self.sample_feat, self.height, self.width)
        self.sample_shared = theano.shared(numpy.zeros(self.sample_shape, dtype=numpy.float32), "shared sample")
        self.sample = self.layers[-1].output[:,self.corner_num:,:,:]

        logging.verbose("Adding", self)

    def parse_desc(layers, name, tags, params):
        if name != "DNC":
            return False

        sample_feat = params.get(0, 512)
        cost_factor = params.get(1, 1.0)
        dropout = params.get(2, 0.0)
        use_center = "C" in tags
        layers.append(DeNetCornerLayer(layers, sample_feat, cost_factor, dropout, use_center))
        return True

    def export_json(self):
        json=super().export_json()
        json.update({"sampleFeat" : self.sample_feat, 
                     "useCenter" : self.use_center, 
                     "costFactor" : self.cost_factor, 
                     "dropout" : self.dropout})
        return json

    def get_target(self, model, samples, metas):

        corner_pr = numpy.zeros(self.corner_shape, dtype=numpy.float32)

        #find all onscreen corners
        for b, meta in enumerate(metas):
            for bbox in meta["bbox"]:

                x0 = int(round(bbox[0]*self.width))
                y0 = int(round(bbox[1]*self.height))
                x1 = max(x0, int(round(bbox[2]*self.width))-1)
                y1 = max(y0, int(round(bbox[3]*self.height))-1)
                x0_valid = (x0 >= 0 and x0 < self.width)
                y0_valid = (y0 >= 0 and y0 < self.height)
                x1_valid = (x1 >= 0 and x1 < self.width)
                y1_valid = (y1 >= 0 and y1 < self.height)
                if x0_valid and y0_valid:
                    corner_pr[b,1,0,y0,x0] = 1.0
                if x1_valid and y0_valid:
                    corner_pr[b,1,1,y0,x1] = 1.0
                if x0_valid and y1_valid:
                    corner_pr[b,1,2,y1,x0] = 1.0
                if x1_valid and y1_valid:
                    corner_pr[b,1,3,y1,x1] = 1.0
                
                #add center point
                if self.use_center:
                    cx = int(round((bbox[0]+bbox[2])*0.5*self.width))
                    cy = int(round((bbox[1]+bbox[3])*0.5*self.height))
                    if cx >= 0 and cx < self.width and cy >= 0 and cy < self.height:
                        corner_pr[b,1,4,cy,cx] = 1.0
                        

        corner_pr[:,0,:,:,:] = 1.0 - corner_pr[:,1,:,:,:]
        corner_pr /= self.width*self.height*self.corner_num

        #apply dropout
        if self.dropout > 0.0:
            mask = numpy.random.binomial(1, 1.0 - self.dropout, (self.corner_shape[0], self.corner_shape[2], self.corner_shape[3], self.corner_shape[4])).astype(numpy.float32)
            corner_pr *= mask[:,None,:,:,:] / (1.0 - self.dropout)

        #pack indices / values
        return numpy.array([], dtype=numpy.int64), corner_pr.flatten()

    #return negative log-likelihood training cost
    def cost(self, yt_index, yt_value):

        #unpack indexs and values
        corner_pr = yt_value.reshape(self.corner_shape)
        corner_cost = -tensor.sum(corner_pr*self.corner_pr, axis=[1,2,3,4]).mean() / math.log(2)

        #debug print costs
        corner_cost = theano.printing.Print('Corner Cost:')(self.cost_factor*corner_cost)
        return corner_cost


#DEBUGGING
if __name__ == '__main__':

    x_shape = [2,2*2*2,8,8]
    x = numpy.zeros(x_shape, dtype=numpy.int32)
    for i in range(x_shape[1]):
        x[:,i,:,:] = i


    #(b,y,x,c)
    y = x.transpose((0,2,3,1))

    #(b,y,x,f,sy,sx)
    y = y.reshape((2, 8, 8, 2, 2, 2))

    #(b,f,x,cx,y,cy)
    y = y.transpose((0,3,2,5,1,4))

    #(b,f,x,cx,y*cy)
    y = y.reshape((2, 2, 8, 2, 8*2))

    #(b,f, y*cy,x,cx)
    y = y.transpose((0,1,4,2,3))

    #(b,f,x,cx,y*cy)
    y = y.reshape((2,2, 8*2, 8*2))

    print(y)

