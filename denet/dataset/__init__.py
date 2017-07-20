import time
import numpy
import math
import random
import fnmatch
import scipy
import json
import threading
from PIL import Image

import denet.common as common
import denet.common.logging as logging

class DatasetExportThread(threading.Thread):
    def __init__(self, model, data, subset, batch_size, training):
        threading.Thread.__init__(self)
        self.model = model
        self.data = data
        self.subset = subset
        self.training = training
        self.batch_size = batch_size
        self.data_export = None
        self.start()

    def run(self):
        logging.info("Exporting subset (%i/%i)"%(self.subset+1, self.data.subset_num))
        timer = common.Timer()
        self.data.load_from_subset(self.subset)
        timer.mark()
        self.data_export = self.data.export(self.batch_size)
        timer.mark()
        logging.info("Finished exporting subset (%i/%i)"%(self.subset+1, self.data.subset_num),"- load took %i sec, export took %i sec"%(timer.delta(0), timer.delta(1)))

    def wait(self):
        self.join()

    def get_export(self):
        return self.data_export

    def get_labels(self):
        return self.data.get_labels()

class DatasetAbstract(object):
    def __init__(self):

        #(fname, x, label, y)
        self.data = []
        self.src_prefix = ""
        self.class_labels = {}
        self.subset_num = 1
        self.subset_index = -1
        self.subset_total_size = 0
        self.subset_size = 0
        self.thread_num = 1
        self.partial_mode = "ignore"
        self.sample_mode = "default"

    #copy dataset
    def copy(self, copy_data=True):

        r = type(self)()
        r.src_prefix = self.src_prefix
        r.class_labels = self.class_labels
        r.subset_num = self.subset_num
        r.subset_index = self.subset_index
        r.subset_total_size = self.subset_total_size
        r.subset_size = self.subset_size
        r.thread_num = self.thread_num
        r.partial_mode = self.partial_mode

        #PIL Image doesnt support copy.copy / deepcopy
        if copy_data:
            if self.get_data_type() == "image":
                r.data = [(fname, d.copy(), meta.copy()) for fname,d,meta in self.data]
            elif self.get_data_type() == "array":
                r.data = [(fname, numpy.copy(d), meta.copy()) for fname,d,meta in self.data]

        return r

    #load data from directory
    def load(self, src_prefix, data_format, class_labels=None):
        raise NotImplementedError()

    #load next dataset subset
    def load_from_subset(self, index):
        pass

    #save dataset
    def save(self, output_prefix):
        raise NotImplementedError()

    #save dataset
    def save_symbolic(self, output_dir):
        raise NotImplementedError()

    def get_subset_size(self, subset = 0):
        if subset == (self.subset_num-1):
            return (self.subset_total_size%self.subset_size)
        else:
            return self.subset_size

    #update dataset with probabilities
    def update(self, pr):
        if self.sample_mode == "confusion":
            self.confusion = numpy.zeros((self.get_class_num(), self.get_class_num()), dtype=numpy.float64)
            predict_cls = numpy.argmax(pr, axis=1)
            for i,t in enumerate(self.data):
                fname,d,meta = t
                self.confusion[meta["class"], predict_cls[i]] += 1

            self.confusion /= numpy.sum(self.confusion, axis=1)[:,None]
            numpy.fill_diagonal(self.confusion, 0.0)

            logging.info("Confusion Matrix:\n", self.confusion)
            logging.info("Error Rates:\n", numpy.sum(self.confusion, axis=1))

        #assign partially labelled items to previous prediction
        if self.partial_mode == "previous" or self.partial_mode == "max":
            cls = numpy.argmax(pr, axis=1)
            for i,t in enumerate(self.data):
                fname,d,meta = t
                if meta["partial"]:
                    meta["class"] = cls[i]
                    self.data[i] = (fname, d, meta)

        #update partially labelled items predicted probabilities
        elif "momentum" in self.partial_mode:
            if "hard" in self.partial_mode:
                cls = numpy.argmax(pr, axis=1)
                pr.fill(0.0)
                pr[numpy.arange(pr.shape[0]), cls] = 1.0

            p = float(self.partial_mode.split(",")[1])
            for i,t in enumerate(self.data):
                fname,d,meta = t
                if meta["partial"]:
                    meta["pr"] = [p*meta["pr"][c] + (1.0 - p)*pr[i,c] for c in range(pr.shape[1])]
                    meta["class"] = meta["pr"].index(max(meta["pr"]))
                    self.data[i] = (fname, d, meta)


    #get length of loaded data
    def __len__(self):
        return len(self.data)

    #get number of samples in all subsets:
    def get_total_size(self):
        return self.subset_total_size

    def get_class_num(self):
        return len(self.class_labels)

    def get_labels(self):
        return [meta["image_class"] for _,_, meta in self.data]

    def get_metas(self):
        return [meta for _,_, meta in self.data]

    #return type of data
    def get_data_type(self):
        if len(self.data) > 0:
            if type(self.data[0][1]) is Image.Image:
                return "image"
            elif type(self.data[0][1]) is numpy.ndarray:
                return "array"

        raise Exception("Cannot get data type!")

    #return dimensions of data
    def get_data_shape(self):
        if len(self.data) > 0:
            if self.get_data_type() == "image":
                if self.data[0][1].mode == "RGB":
                    return (3, self.data[0][1].size[0], self.data[0][1].size[1])
                else:
                    return (1, self.data[0][1].size[0], self.data[0][1].size[1])

            elif self.get_data_type() == "array":
                return self.data[0][1].shape
        else:
            raise Exception("Cannot get data shape! Please override get_data_shape() in Dataset class.")

    #split dataset into N folds within each class
    def split_folds(self, nfolds):
        data_folds=[self.copy(False) for _ in range(nfolds)]
        for i,d in enumerate(self.data):
            data_folds[i%nfolds].data.append(d)

        return data_folds

    #combine two datasets
    def concatenate(self, data):
        r = self.copy(True)
        r.data += data.data
        return r

    #randomly sort data
    def shuffle(self, mode="random"):
        if mode == "random":
            random.shuffle(self.data)
        else:
            raise Exception("Unknown shuffle mode:", mode)

    #returns new dataset with images converted to given PIL Image mode
    def set_image_mode(self, mode):
        assert self.get_data_type() == "image"
        new_data=[]
        for fname, im, meta in self.data:
            new_data.append((fname, im.convert(mode, dither=None), meta))
        self.data = new_data

    #returns new dataset with bordered image
    def add_border(self, n):
        new_data = []
        for fname, im, meta in self.data:
            if self.get_data_type() == "image":
                r = Image.new(im.mode, (im.size[0] + 2*n, im.size[1] + 2*n))
                r.paste(im, (n, n))
            elif self.get_data_type() == "array":
                r = numpy.zeros((im.shape[0], im.shape[1] + 2*n, im.shape[2] + 2*n))
                r[:, n:(r.shape[1]-n), n:(r.shape[2]-n)] = im

            new_data.append((fname, r, meta))
        self.data = new_data

    #returns new dataset with images resized
    def resize(self, size, filter=Image.BILINEAR):
        new_data = []
        for fname, im, meta in self.data:
            if self.get_data_type() == "image":
                new_data.append((fname, im.resize(size, filter), meta))

            elif self.get_data_type() == "array":
                zx = size[0] / im.shape[1]
                zy = size[1] / im.shape[2]
                r = scipy.ndimage.interpolation.zoom(im[0,:,:], [zx,zy])
                g = scipy.ndimage.interpolation.zoom(im[1,:,:], [zx,zy])
                b = scipy.ndimage.interpolation.zoom(im[2,:,:], [zx,zy])
                new_data.append((fname, numpy.concatenate((r[None,:,:],g[None,:,:],b[None,:,:]), axis=0), meta))

        self.data = new_data

    #
    def transform_sample(self, sample, rotate=0, scale=[1,1], shear=[0,0], offset=[0,0], mirror = False, bilinear=False):

        fname,im,meta = sample
        rot = numpy.matrix([[math.cos(rotate), math.sin(rotate)], [-math.sin(rotate), math.cos(rotate)]])
        scale = numpy.matrix([[1.0/scale[0], 0], [0, 1.0 / scale[1]]])
        shear_x = numpy.matrix([[1,shear[0]], [0,1]])
        shear_y = numpy.matrix([[1,0], [shear[1],1]])
        tr = rot*scale*shear_x*shear_y
        if self.get_data_type() == "image":

            mode = Image.BILINEAR if bilinear else Image.NEAREST
            center = (im.size[0]/2, im.size[1]/2)
            ox = center[0]-center[0]*tr[0,0]-center[1]*tr[0,1] - offset[0]
            oy = center[1]-center[0]*tr[1,0]-center[1]*tr[1,1] - offset[1]
            im = im.transform(im.size, Image.AFFINE, (tr[0,0],tr[0,1],ox,tr[1,0],tr[1,1],oy), resample=resample)
            if mirror:
                im = ImageOps.mirror(im)

        elif self.get_data_type() == "array":

            order = 1 if bilinear else 0
            center=(im.shape[1]/2, im.shape[2]/2)
            ox=center[0] - center[0]*tr[0,0] - center[1]*tr[0,1] - offset[0]
            oy=center[1] - center[0]*tr[1,0] - center[1]*tr[1,1] - offset[1]
            r = numpy.zeros_like(im)
            r[0,:,:] = scipy.ndimage.interpolation.affine_transform(im[0,:,:], tr[0:2,0:2], [ox,oy], order=order)
            r[1,:,:] = scipy.ndimage.interpolation.affine_transform(im[1,:,:], tr[0:2,0:2], [ox,oy], order=order)
            r[2,:,:] = scipy.ndimage.interpolation.affine_transform(im[2,:,:], tr[0:2,0:2], [ox,oy], order=order)

            if mirror:
                r = r[:,:,::-1]
            im = r

        return (fname, im, meta)

    #random affine distorted images (destructive)
    def distort_affine(self, dtheta=0, dscale=0, dshear=0, doffset=0, mirror=0, use_integer_offset=False, bilinear=False):

        class AffineDistortWorker(threading.Thread):
            def __init__(self, data, index, theta, scale, shear, offset, mirror, bilinear):
                threading.Thread.__init__(self)
                self.data = data
                self.index = index
                self.theta = theta
                self.scale = scale
                self.offset = offset
                self.shear = shear
                self.mirror = mirror
                self.bilinear = bilinear

            def run(self):
                self.data.data[self.index] = self.data.transform_sample(self.data.data[self.index], self.theta, self.scale, self.shear, self.offset, self.mirror, self.bilinear)

        index=0
        while index < len(self.data):

            #start threads
            active_workers=[]
            while len(active_workers) < self.thread_num and index < len(self.data):
                theta = numpy.random.uniform(-dtheta, dtheta)*math.pi/180.0
                scale = numpy.random.uniform(1.0 - dscale, 1.0 + dscale)

                if use_integer_offset:
                    tx = numpy.random.random_integers(-doffset, doffset)
                    ty = numpy.random.random_integers(-doffset, doffset)
                else:
                    tx = numpy.random.uniform(-doffset, doffset)
                    ty = numpy.random.uniform(-doffset, doffset)

                kx = numpy.random.uniform(-dshear, dshear)
                ky = numpy.random.uniform(-dshear, dshear)
                m = (numpy.random.random() < mirror)

                worker = AffineDistortWorker(self, index, theta, (scale, scale), (kx,ky), (tx,ty), m, bilinear)
                worker.start()
                active_workers.append(worker)
                index += 1

            #wait for workers
            for worker in active_workers:
                worker.join()

    def transform(self, rotate=0, scale=[1,1], offset=[0,0], mirror = False,  shear=[0,0], super_sample = 1, resample=Image.BILINEAR):
        distort = self.copy(False)
        distort.data = [self.apply_transform(v, rotate, scale, offset, mirror, shear, super_sample, resample) for v in self.data]
        return distort

    #adding horizontally mirrored data
    def augment_mirror(self):
        if self.get_data_type() == "image":
            self.data += [(fname, im.transpose(Image.FLIP_LEFT_RIGHT), meta) for fname,im,meta in self.data]
        elif self.get_data_type() == "array":
            self.data += [(fname, d[:,:,::-1], meta) for fname,d,meta in self.data]

    #call to set data with modified data
    def set_data(self, data):
        data_list = []
        for fname,d,meta in data:
            if self.partial_mode == "ignore" and meta.get("partial", True):
                continue
            data_list.append((fname,d,meta))

        self.data = data_list

    #export dataset for training / testing
    def export(self, batch_size=1, dtype=numpy.float32):
        size = batch_size * math.ceil(len(self) / batch_size)
        data_x = numpy.zeros((size, self.get_data_shape()[0], self.get_data_shape()[1], self.get_data_shape()[2]), dtype=dtype)
        data_meta=[]
        for i in range(size):
            index=i if (i < len(self.data)) else random.randint(0, len(self.data)-1)
            fname, im_x, meta = self.data[index]

            #(H,W,C) -> (C,W,H) -> (C,H,W)
            if self.get_data_type() == "image":
                im_x = numpy.array(im_x, dtype=dtype)/255.0
                im_x = numpy.swapaxes(im_x, 0, 2)
                im_x = numpy.swapaxes(im_x, 1, 2)

            data_x[i, ...] = im_x[...]
            data_meta.append(meta)

        return (data_x, data_meta, len(self.data))

#helper function for loading datasets automatically determining type
def load(src_prefix, data_format, is_training=False, thread_num=1, class_labels=None):
    from .basic import DatasetFromDir, DatasetFromArray
    from .imagenet import DatasetImagenet
    from .mscoco import DatasetMSCOCO
    from .pascal_voc import DatasetPascalVOC

    if "imagenet" in data_format:
        data = DatasetImagenet()
    elif "mscoco" in data_format:
        data = DatasetMSCOCO()
    elif "voc" in data_format:
        data = DatasetPascalVOC()
    elif data_format == "npy" or data_format == "npz":
        data = DatasetFromArray()
    else:
        data = DatasetFromDir()

    data.load(src_prefix, data_format, is_training, thread_num, class_labels)
    return data

