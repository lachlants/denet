import os
import sys
import math
import random
import xml.etree.ElementTree as xml
import numpy

import denet.common as common
import denet.common.logging as logging
from denet.dataset import DatasetAbstract
from denet.dataset.image_loader import ImageLoader

class DatasetPascalVOC(DatasetAbstract):

    def get_data_shape(self):
        return (3, self.output_size, self.output_size)

    def copy(self, copy_data=True):
        r = super().copy(copy_data)
        r.images = self.images
        r.scale = self.scale
        r.crop = self.crop
        r.is_training = self.is_training
        r.augment_resnet = self.augment_resnet
        r.augment_lenet = self.augment_lenet
        r.augment_mirror = self.augment_mirror
        r.augment_photometric = self.augment_photometric
        return r

    def shuffle(self, mode="random"):
        random.shuffle(self.images)

    def load_from_subset(self, subset):

        # if self.subset_index == subset:
        #     return

        index_start = subset*self.subset_size
        index_end = min((subset+1)*self.subset_size, self.subset_total_size)

        logging.info("Loading from subset %i / %i (%i threads, %i start, %i end)"%(subset, self.subset_num, self.thread_num, index_start, index_end))
        self.data = self.image_loader.load(self.images[index_start:index_end])
        self.subset_index = subset

    def load(self, input_dir, data_format, is_training, thread_num, class_labels=None):

        self.thread_num = thread_num

        #get arguments
        param_str = ",".join(data_format.split(",")[1:])
        format_params = common.get_params_dict(param_str)

        #Hard coded class labels
        self.class_labels = {"aeroplane":0, "bicycle":1, "bird":2, "boat":3, "bottle":4, "bus":5, "car":6, "cat":7, "chair":8, "cow":9, "diningtable":10,
                             "dog":11, "horse":12, "motorbike":13, "person":14, "pottedplant":15, "sheep":16, "sofa":17, "train":18, "tvmonitor":19}

        #select datasets to include

        def get_files(data_set, image_set):
            logging.info("Loading pascal %s %s..."%(data_set, image_set))
            with open(os.path.join(input_dir, "%s/ImageSets/Main/%s.txt"%(data_set,image_set)), "r") as f:
                fnames = [os.path.join(input_dir, "%s/JPEGImages/%s.jpg"%(data_set,index.rstrip())) for index in f.readlines()]
            return fnames

        #VOC 2007
        files=[]
        param = [s for s in format_params.keys() if s.startswith("2007")]
        param = param[0] if len(param)>0 else ""
        if "train" in param:
            files += get_files("VOC2007", "train")
        if "val" in param:
            files += get_files("VOC2007", "val")
        if "test" in param:
            files += get_files("VOC2007", "test")

        #VOC 2012
        param = [s for s in format_params.keys() if s.startswith("2012")]
        param = param[0] if len(param)>0 else ""
        if "train" in param:
            files += get_files("VOC2012", "train")
        if "val" in param:
            files += get_files("VOC2012", "val")
        if "test" in param:
            files += get_files("VOC2012", "test")

        logging.info("Finding images / metadata")
        self.images=[]
        for fname in files:

            bboxs = []
            difficult=[]

            #load objects associated with image
            anno_dir = os.path.join(os.path.dirname(os.path.dirname(fname)), "Annotations")
            obj_fname = os.path.join(anno_dir, os.path.splitext(os.path.basename(fname))[0] + ".xml")
            if os.path.isfile(obj_fname):
                obj_tree = xml.parse(obj_fname).getroot()
                for obj in obj_tree.iter("object"):
                    cls = self.class_labels[obj.find("name").text]

                    diff = bool(int(obj.find("difficult").text) > 0)
                    difficult.append(diff)

                    #minus one due to MATLAB stupidity
                    bndbox = obj.find("bndbox")
                    min_x = int(bndbox.find("xmin").text) - 1
                    min_y = int(bndbox.find("ymin").text) - 1
                    max_x = int(bndbox.find("xmax").text) - 1
                    max_y = int(bndbox.find("ymax").text) - 1
                    bboxs.append((cls, (min_x,min_y,max_x,max_y)))

            elif is_training:
                raise Exception("Could not find annotations for training data!")

            self.images.append({"fname" : fname, "bboxs": bboxs, "difficult": difficult})

        #sort images initially
        self.images.sort(key=lambda im: im["fname"])
        self.image_loader = ImageLoader(thread_num, is_training, format_params)
        # self.image_loader.rgb_mean = numpy.array([0.41, 0.46, 0.48], dtype=numpy.float32)
        # self.image_loader.rgb_std = numpy.array([1,1,1], dtype=numpy.float32)

        #from Imagenet (natural image set = should have similar values)
        self.image_loader.rgb_mean = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
        self.image_loader.rgb_std = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)
        self.image_loader.rgb_eigen_val = numpy.array([0.2175,  0.0188, 0.0045], dtype=numpy.float32)
        self.image_loader.rgb_eigen_vec = numpy.array([[-0.5675,  0.7192,  0.4009],
                                                       [-0.5808, -0.0045, -0.8140],
                                                       [-0.5836, -0.6948,  0.4203]], dtype=numpy.float32)


        #subset
        self.output_size = self.image_loader.crop
        self.subset_size = min(format_params.get("images_per_subset", 10000), len(self.images))
        self.subset_total_size = len(self.images)
        self.subset_num = format_params.get("subset_num", sys.maxsize)
        self.subset_num = min(self.subset_num, int(math.ceil(self.subset_total_size / self.subset_size)))
        self.subset_index = -1

        logging.info("Using Pascal VOC dataset - size:", self.subset_total_size, "subset_num", self.subset_num, "images per subset:", self.subset_size, self.image_loader)

    #export detection in pascal VOC format
    def export_detections(output_dir, detections, width, height, class_labels_inv):

        output={}
        for i,r in enumerate(detections):

            meta = r["meta"]
            image_id = os.path.splitext(os.path.basename(meta["image"]["fname"]))[0]
            scale_x, scale_y = meta["scale"]
            offset_x, offset_y = meta["offset"]
            image_width, image_height = meta["image_size"]

            for pr,cls,bbox in r["detections"]:
                x0 = max(min(int((bbox[0]*width + offset_x) / scale_x) + 1, image_width), 1)
                y0 = max(min(int((bbox[1]*height + offset_y) / scale_y) + 1, image_height), 1)
                x1 = max(min(int((bbox[2]*width + offset_x) / scale_x) + 1, image_width), 1)
                y1 = max(min(int((bbox[3]*height + offset_y) / scale_y) + 1, image_height), 1)
                if cls not in output:
                    output[cls] = []
                output[cls].append((image_id, pr,x0,y0,x1,y1))

        for cls in output.keys():
            fname = os.path.join(output_dir, "comp4_det_test_%s.txt"%class_labels_inv[cls])
            with open(fname, "w") as f:
                for d in output[cls]:
                    f.write("%s %0.6f %.6f %.6f %.6f %.6f\n"%d)

    #compute average precision for each class using pascal voc metric
    def get_precision(detections, overlap_threshold = 0.5):

        #inverse class labelling
        class_labels_inv = {0:"aeroplane", 1:"bicycle", 2:"bird", 3:"boat", 4:"bottle", 5:"bus", 6:"car", 7:"cat", 8:"chair", 9:"cow", 10:"diningtable",
                            11:"dog", 12:"horse", 13:"motorbike", 14:"person", 15:"pottedplant", 16:"sheep", 17:"sofa", 18:"train", 19:"tvmonitor"}
        
        coverage=0
        coverage_total=0
        for r in detections:
            for cls_a, bbox_a in zip(r["meta"]["class"], r["meta"]["bbox"]):
                coverage_total += 1
                for _,cls_b,bbox_b in r["detections"]:
                    if cls_a == cls_b and common.overlap_iou(bbox_a, bbox_b) > overlap_threshold:
                        coverage += 1
                        break

        logging.info("coverage: %.2f%% (%i,%i)"%(100.0*coverage/coverage_total, coverage, coverage_total))

        #collect all detections and groundtruth detections into classes
        gts_cls = [[] for _ in range(20)]
        dts_cls = [[] for _ in range(20)]
        for image_id,r in enumerate(detections):
            for pr,cls,bbox in r["detections"]:
                dts_cls[cls].append((image_id, pr, bbox))

            for cls, bbox, difficult in zip(r["meta"]["class"], r["meta"]["bbox"], r["meta"]["image"]["difficult"]):
                gts_cls[cls].append((image_id, difficult, bbox))

        logging.warning("WARNING: does not obtain exact results relative to VOCcode implementation!")
        logging.info("Using overlap threshold: %.2f"%overlap_threshold)
        mean_ap=0
        for cls in range(20):
            gts = gts_cls[cls]
            dts = dts_cls[cls]

            non_difficult_num = 0
            for _, diff, _ in gts:
                if not diff:
                    non_difficult_num += 1

            #sort by confidence
            dts.sort(key=lambda d:-d[1])
            tp = numpy.zeros((len(dts),), dtype=numpy.int64)
            fp = numpy.zeros((len(dts),), dtype=numpy.int64)
            gt_found=[]
            for d in range(len(dts)):

                image_id, pr, bbox = dts[d]
                overlap_max = 0
                overlap_index = 0
                for gt_i, gt in enumerate(gts):
                    gt_image_id, _, gt_bbox = gt
                    if gt_image_id == image_id:
                        overlap = common.overlap_iou(bbox, gt_bbox)
                        if overlap > overlap_max:
                            overlap_max = overlap
                            overlap_index = gt_i

                if overlap_max >= overlap_threshold:
                    if not gts[overlap_index][1]:
                        if overlap_index in gt_found:
                            fp[d] = 1
                        else:
                            gt_found.append(overlap_index)
                            tp[d] = 1
                else:
                    fp[d] = 1

            tp = numpy.cumsum(tp)
            fp = numpy.cumsum(fp)
            recall = tp / non_difficult_num
            prec = tp / (tp + fp)

            #VOC 2007 algorithm!
            ap=0
            for t in numpy.linspace(0.0, 1.0, 11):
                n = (recall >= t)
                p = prec[n].max()if n.any() else 0.0
                ap += p / 11

            # #fancy pascal VOC 2012 AP calculation
            # mrec = numpy.array([0.0] + recall.tolist() + [1.0])
            # mpre = numpy.array([0] + prec.tolist() + [0.0])
            # for i in range(mpre.shape[0] - 2, -1, -1):
            #     mpre[i] = max(mpre[i], mpre[i+1])

            # ap=0
            # for i in range(mrec.shape[0]-1):
            #     if mrec[i+1] != mrec[i]:
            #         ap += (mrec[i+1] - mrec[i]) * mpre[i+1]

            mean_ap += ap
            logging.info("%s - AP: %.4f (%i detections,  %i groundtruth, %i non difficult)"%(class_labels_inv[cls], ap, len(dts), len(gts), non_difficult_num))

        mean_ap /= 20
        logging.info("Mean AP: %.4f"%mean_ap)
