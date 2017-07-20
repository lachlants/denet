import os
import sys
import time
import numpy
import math
import random
import copy
import json
import tarfile
import xml.etree.ElementTree as xml

import denet.common as common
import denet.common.logging as logging
from denet.dataset import DatasetAbstract
from denet.dataset.image_loader import ImageLoader

class DatasetImagenet(DatasetAbstract):

    def copy(self, copy_data=True):
        r = super().copy(copy_data)
        r.images = self.images
        r.image_loader = self.image_loader
        return r

    def shuffle(self, mode="random"):
        random.shuffle(self.images)

    def load_from_subset(self, subset):
        if self.subset_index == subset:
            return

        logging.info("Loading from subset %i / %i (%i threads)"%(subset, self.subset_num, self.thread_num))

        index_start = subset*self.subset_size
        index_end = min((subset+1)*self.subset_size, self.subset_total_size)
        self.data = self.image_loader.load(self.images[index_start:index_end])
        self.subset_index = subset

    def load(self, input_dir, data_format, is_training, thread_num, class_labels=None):

        from .basic import DatasetFromDir

        self.input_dir = input_dir
        if self.input_dir[-1] == '/':
            self.input_dir = self.input_dir[:-1]

        self.data_format = data_format
        self.thread_num = thread_num

        #generate class labels
        self.class_labels = class_labels

        fname = os.path.join(os.path.dirname(self.input_dir), "class_labels.txt")
        if os.path.isfile(fname) and self.class_labels is None:
            logging.info("Loading class labels from:", fname)
            self.class_labels = {}
            with open(fname, "r") as f:
                for line in f.readlines():
                    tokens = line.rstrip('\n').split(" ")
                    self.class_labels[tokens[1]] = int(tokens[0])

        elif self.class_labels is None:
            self.class_labels = DatasetFromDir.find_class_labels(input_dir)

        #check to see if buffered file list is present
        list_fname = os.path.join(input_dir, "image_list.json")
        if os.path.isfile(list_fname):
            logging.info("Loading dataset metadata:", list_fname)
            json_data = common.json_from_file(list_fname)
            if json_data.get("version", 0) < 1:
                logging.warning("Warning: image_list.json is old version, missing bounding boxs!")
                self.images= [{"fname":fname, "bboxs":[]} for fname in json_data["images"]]
            else:
                self.images = json_data["images"]
        else:

            bbox_dir = os.path.join(os.path.dirname(input_dir), "bbox")
            if not os.path.isdir(bbox_dir):
                raise Exception("ERROR: cannot find bbox dir:" + bbox_dir)

            fnames=[]
            for i,c in enumerate(os.listdir(input_dir).sort()):
                images_cls = DatasetFromDir.find_paths(os.path.join(input_dir, c), "*.JPEG")
                logging.info("Found %i images for class"%len(images_cls), c)
                fnames += images_cls

            logging.info("Finding bboxs in:", bbox_dir)
            self.images=[]
            for i,fname in enumerate(fnames):
                logging.verbose("%i/%i"%(i, len(fnames)))
                cls_name = os.path.basename(os.path.dirname(fname))
                obj_fname = os.path.join(bbox_dir, cls_name, os.path.splitext(os.path.basename(fname))[0] + ".xml")
                bboxs=[]
                if os.path.isfile(obj_fname):
                    obj_tree = xml.parse(obj_fname).getroot()
                    size = obj_tree.find("size")
                    width = int(size.find("width").text)
                    height = int(size.find("height").text)
                    for obj in obj_tree.iter("object"):
                        bndbox = obj.find("bndbox")
                        min_x = int(bndbox.find("xmin").text)
                        min_y = int(bndbox.find("ymin").text)
                        max_x = int(bndbox.find("xmax").text)
                        max_y = int(bndbox.find("ymax").text)
                        bboxs.append({"x0":min_x, "x1":max_x, "y0":min_y, "y1":max_y})

                self.images.append({"fname":fname, "bboxs":bboxs})

            try:
                logging.info("Saving dataset metadata:", list_fname)
                common.json_to_file(list_fname, {"images":self.images, "version":1})
            except Exception as e:
                logging.warning("Warning: failed to write buffered image list - ", e)

        #add/fix fields to fit new image_loader interface
        for image in self.images:
            fname = image["fname"]
            cls = self.class_labels[os.path.basename(os.path.dirname(fname))]
            image["class"] = cls
            image["bboxs"] = [(cls, (bb["x0"], bb["y0"], bb["x1"], bb["y1"])) for bb in image["bboxs"]]

        param_str = ",".join(data_format.split(",")[1:])
        format_params = common.get_params_dict(param_str)
        self.image_loader = ImageLoader(thread_num, is_training, format_params)

        #from facebook resnet implementation
        self.image_loader.rgb_mean = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
        self.image_loader.rgb_std = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)
        self.image_loader.rgb_eigen_val = numpy.array([0.2175,  0.0188, 0.0045], dtype=numpy.float32)
        self.image_loader.rgb_eigen_vec = numpy.array([[-0.5675,  0.7192,  0.4009],
                                                       [-0.5808, -0.0045, -0.8140],
                                                       [-0.5836, -0.6948,  0.4203]], dtype=numpy.float32)

        #others
        self.subset_size= format_params.get("images_per_subset", 10000)
        self.use_null_class = format_params.get("null", False)
        self.subset_num = format_params.get("subset_num", sys.maxsize)
        self.bbox_only = format_params.get("bbox_only", False)

        #only use samples with bounding boxes
        if self.image_loader.is_training and self.bbox_only:
            images_bbox=[]
            for image in self.images:
                if len(image["bboxs"]) > 0:
                    images_bbox.append(image)
            self.images = images_bbox

        #append null class
        if self.use_null_class and not "null" in self.class_labels:
            self.class_labels["null"] = len(self.class_labels)

        self.subset_index = -1
        self.subset_total_size = len(self.images)
        self.subset_num = min(self.subset_num, int(math.ceil(self.subset_total_size / self.subset_size)))

        logging.info("Using Imagenet dataset - size:", self.subset_total_size, "subset_num", self.subset_num, "images per subset:", self.subset_size, self.image_loader)

    #compute error
    def get_localization_error(detections):
        error=0
        det_truth=0
        det_total=0
        for d in detections:
            meta = d["meta"]
            dets = d["detections"]
            dets.sort(key=lambda t:t[0])

            det_truth += len(meta["class"])
            det_total += len(dets)

            positive = False
            for _, cls_a, bbox_a in dets[:min(5,len(dets))]:
                for cls_b, bbox_b in zip(meta["class"], meta["objs"]):
                    if cls_a == cls_b and common.overlap_iou(bbox_a, bbox_b) > 0.5:
                        positive = True

            if not positive:
                error += 1

        logging.info("Imagenet localization error: %.2f (%i images, %i true detections, %i total detections)"%(100.0*error / len(detections), len(detections), det_truth, det_total))
