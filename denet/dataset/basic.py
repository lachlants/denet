import os
import sys
import time
import numpy
import random
import fnmatch
import copy
from PIL import Image

import denet.common.logging as logging
from denet.dataset import DatasetAbstract

class DatasetFromDir(DatasetAbstract):
    def find_class_labels(src_dir):
        labels=dict()
        classes=os.listdir(src_dir)
        for c in classes:
            if os.path.isdir(os.path.join(src_dir, c)) and not c in labels:
                labels[c]=len(labels)

        return labels

    def find_paths(directory, pattern):
        paths = []
        for root, dirs, files in os.walk(directory, topdown=False, followlinks=True):
            for basename in files:
                filename = os.path.join(root, basename)
                if fnmatch.fnmatch(filename, pattern):
                    paths.append(filename)
        paths.sort()
        return paths

    #load from directory of classes
    def load(self, input_dir, ext, is_training, thread_num, class_labels=None):
        self.class_labels = class_labels

        #generate class labels
        if self.class_labels is None:
            self.class_labels = DatasetFromDir.find_class_labels(input_dir)

        #load images
        classes = os.listdir(input_dir)
        for c in classes:
            cls = self.class_labels[c]
            images = DatasetFromDir.find_paths(os.path.join(input_dir, c), "*." + ext)

            logging.info("Found class " + c + " (" + str(cls) + ") with " + str(len(images)) + " images")
            for f in images:
                imfile = Image.open(f)
                basename = f.replace(input_dir, "")
                self.data.append((basename, imfile.copy(), {"image_class":cls, "partial":False}))

        #sort by class
        self.data.sort(key=lambda d:d[2]["image_class"])
        logging.info("Loaded " + str(len(self)) + " Samples")

class DatasetFromArray(DatasetAbstract):
    def load(self, src_prefix, ext, is_training, thread_num, class_labels=None):

        data = numpy.load(os.path.join(src_prefix, "_data.npy"))
        labels = numpy.load(os.path.join(src_prefix, "_labels.npy"))

        if class_labels is None:
            label_min = int(labels.min())
            label_max = int(labels.max())
            self.class_labels = {}
            for i in range(label_min, label_max + 1):
                self.class_labels.update({ str(i) : len(self.class_labels)})
        else:
            self.class_labels = class_labels

        self.data_type = "array"
        self.data=[]
        for i in range(data.shape[0]):
            cls = self.class_labels[str(int(labels[i]))]
            self.data.append((data_fname, numpy.array(data[i,:,:,:], dtype=numpy.float32, copy=True), {"class":cls, "partial":False}))

        del data, labels
