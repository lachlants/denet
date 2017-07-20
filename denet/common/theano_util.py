import os
import numpy
import theano
import theano.tensor as tensor
import denet.common.logging as logging


def export_graph(fname, func):
    logging.verbose("Saving function graph: " + fname)
    with open(fname, "w") as f:
        theano.printing.debugprint(func, file=f, print_type=True)

def profile(func, it_num, *args):
    assert os.environ.get('CUDA_LAUNCH_BLOCKING', '0') == '1', "Requires CUDA_LAUNCH_BLOCKING=1 to get proper results"
    assert not func.profile is None, "Compile function with profile=True"

    logging.info("Profiling function (%i it)"%it_num)
    for _ in range(it_num):
        func(*args)

    try:
        func.profile.summary()
    except:
        pass

#log softmax along specified axis
def log_softmax(x, axis):
    xdev = x - x.max(axis=axis, keepdims=True)
    return xdev - tensor.log(tensor.sum(tensor.exp(xdev), axis=axis, keepdims=True))

#smooth L1 loss
def smooth_L1(x):
    x_abs = abs(x)
    return tensor.switch(x_abs < 1, 0.5*tensor.pow(x, 2), x_abs - 0.5)


overlap_func=None
def get_overlap_iou(obj_bboxs, sample_bboxs):
    
    global overlap_func
    if overlap_func is None:
        logging.debug("Building overlap function")
        x_bboxs = tensor.matrix()
        y_bboxs = tensor.matrix()
        x_area = (x_bboxs[:,2] - x_bboxs[:,0])*(x_bboxs[:,3] - x_bboxs[:,1])
        y_area = (y_bboxs[:,2] - y_bboxs[:,0])*(y_bboxs[:,3] - y_bboxs[:,1])
        dx = tensor.maximum(tensor.minimum(x_bboxs[:,None,2], y_bboxs[None,:,2]) - tensor.maximum(x_bboxs[:,None,0], y_bboxs[None,:,0]), 0)
        dy = tensor.maximum(tensor.minimum(x_bboxs[:,None,3], y_bboxs[None,:,3]) - tensor.maximum(x_bboxs[:,None,1], y_bboxs[None,:,1]), 0)
        area_intersect = dx*dy
        area_union = (x_area[:,None] + y_area[None,:] - area_intersect)
        area_iou = area_intersect / area_union
        overlap_func = theano.function([x_bboxs, y_bboxs], area_iou, allow_input_downcast=True)

    if len(obj_bboxs) == 0 or len(sample_bboxs) == 0:
        return None
    else:
        x = numpy.array(obj_bboxs, dtype=numpy.float32)
        y = numpy.array(sample_bboxs, dtype=numpy.float32)
        return overlap_func(x,y)

#replace all inf/nans in x with v
def replace_inf_nan(x, v):
    return tensor.switch(tensor.or_(tensor.isnan(x), tensor.isinf(x)), v, x)

#apply r = x + delta if r is not inf / nan, else return x
def update_inf_nan(x, delta, v):
    r = x + delta
    return tensor.switch(tensor.or_(tensor.isnan(r), tensor.isinf(r)), x, r)

#will check if shuffle is needed
def dimshuffle(input, s):

    need_shuffle=False
    for i,ss in enumerate(s):
        if i != ss:
            need_shuffle=True
            break

    return input.dimshuffle(s) if need_shuffle else input

#return axes for <input> NOT specified in <axes>
def other_axes(input, axes):
    other_axes=[]
    for a in range(input.ndim):
        if a not in axes:
            other_axes.append(a)
    return other_axes
