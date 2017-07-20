import sys
import gzip
import numpy
import base64
import json
import io
import os
import math
import fnmatch
import time

from denet.common.json_util import *
from denet.common.image_util import *

#
class Timer:
    def __init__(self):
        self.reset()

    def mark(self):
        self.marks.append(time.time())

    def reset(self):
        self.marks = [ time.time() ]

    #time elapsed (sec)
    def current(self):
        return time.time() - self.marks[0]

    #time elapsed (millisec)
    def current_ms(self):
        return 1000*self.current()

    def delta(self, key0, key1=None):
        if key1 is None:
            key1 = key0 + 1
        return self.marks[key1] - self.marks[key0]

    def delta_ms(self, key0, key1=None):
        return 1000*self.delta(key0, key1)

    def deltas(self):
        return [(self.marks[i+1] - self.marks[i]) for i in range(len(self.marks)-1)]

    def deltas_ms(self):
        return [1000*(self.marks[i+1] - self.marks[i]) for i in range(len(self.marks)-1)]

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def find_files(directory, pattern):
    paths = []
    for root, dirs, files in os.walk(directory, topdown=False, followlinks=True):
        for basename in files:
            filename = os.path.join(root, basename)
            if fnmatch.fnmatch(filename, pattern):
                paths.append(filename)

    paths.sort()
    return paths


#search list of layers
def find_layers(layers, layer_names, warn_missing=False):
    if type(layer_names) is str:
        layer_names = [layer_names]

    found_layers = [None]*len(layer_names)
    for layer in layers:
        for i,layer_name in enumerate(layer_names):
            if found_layers[i] is None and layer.type_name == layer_name:
                found_layers[i] = layer

    if warn_missing:
        missed_names=[]
        for i,found_layer in enumerate(found_layers):
            if found_layer is None:
                missed_names.append(layer_names[i])
        if len(missed_names) > 0:
            raise Exception("Could not find layers of name: ", missed_names)

    if len(layer_names) == 1:
        return found_layers[0]
    else:
        return found_layers



#area of intersection between two bboxs (x_min, y_min, x_max, y_max)
def overlap(bbox0, bbox1=(0,0,1,1)):
    dx = max(0, min(bbox0[2], bbox1[2]) - max(bbox0[0], bbox1[0]))
    dy = max(0, min(bbox0[3], bbox1[3]) - max(bbox0[1], bbox1[1]))
    return dx*dy

#relative overlap of bbox0
def overlap_rel(bbox0, bbox1 = (0,0,1,1)):
    a = (bbox0[2] - bbox0[0])*(bbox0[3] - bbox0[1])
    dx = max(0, min(bbox0[2], bbox1[2]) - max(bbox0[0], bbox1[0]))
    dy = max(0, min(bbox0[3], bbox1[3]) - max(bbox0[1], bbox1[1]))
    return dx*dy / a if a > 0 else 0.0

#Imagenet defined overlap ratio = area of intersection / area of union
def overlap_iou(bbox0, bbox1 = (0,0,1,1)):
    a0 = (bbox0[2] - bbox0[0])*(bbox0[3] - bbox0[1])
    a1 = (bbox1[2] - bbox1[0])*(bbox1[3] - bbox1[1])
    ai = overlap(bbox0, bbox1)
    au = a0 + a1 - ai
    return ai / au

#
def clip(x, x_min=None, x_max=None):
    if x_min is None:
        return min(x, x_max)
    elif x_max is None:
        return max(x, x_min)
    else:
        return min(x_max, max(x, x_min))

#
def get_flat_index(stride, *args):
    return sum([x*s for x,s in zip(args, stride)])

#
def ndarray_unpack(v, shapes):

    index=0
    r=[]
    for shape in shapes:
        size = numpy.prod(shape)
        r.append(v[index:(index+size)].reshape(shape))
        index += size
    return r

#get 2d block view of matrix
def block_view_2d(x, block):
    shape= (x.shape[0]/ block[0], x.shape[1]/ block[1]) + block
    strides= (block[0]*x.strides[0], block[1]*x.strides[1]) + x.strides
    return numpy.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

#convert to float or int
def convert_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
#theano
def merge_axes(x, x_shape, axes):
    assert(len(axes) > 1)

    axes.sort()

    non_axes=[]
    for i in range(len(x_shape)):
        if i not in axes:
            non_axes.append(i)

    size = numpy.prod([x_shape[axis] for axis in axes])

    y = x.dimshuffle(tuple(not_axes + axes))
    y = y.flatten(len(non_axes) + 1)

    shuffle = list(range(len(non_axes)))
    shuffle.insert(axes[0], len(non_axes)+1)
    y = y.dimshuffle(shuffle)
    return y

def import_c(fname):
    # import logging
    # logging.getLogger("theano.gof.cmodule").setLevel(logging.DEBUG)

    #compile code
    import theano
    import theano.gof.cmodule as cmodule
    import theano.gof.compilelock as compilelock
    from theano.gof.utils import hash_from_code
    compilelock.get_lock()
    try:
        compile_args = ["-std=c++11", "-fPIC", "-O3", "-fno-math-errno", "-Wno-unused-label", "-Wno-unused-variable", "-Wno-write-strings"]
        compile_dir = cmodule.dlimport_workdir(theano.config.compiledir)
        with open(fname, "r") as f:
            compile_code = f.read()

        compile_hash = os.path.basename(os.path.splitext(fname)[0])
        cmodule_import = cmodule.GCC_compiler.compile_str(compile_hash, compile_code, compile_dir, preargs = compile_args)
    except Exception as e:
        print("Warning: Failed to compile cmodule:", e)
        cmodule_import = None
    finally:
        compilelock.release_lock()

    return cmodule_import

#convert string of parameters into dict 
#format: <param0>=<value0>,<param1>=<value1>
#if no value is provided value is set to True 
def get_params_dict(params):
    format_params={}
    for str in params.split(","):
        pv = str.split("=")
        param = pv[0]
        value = True if len(pv) == 1 else convert_num(pv[1])
        format_params[param] = value
        
    return format_params
