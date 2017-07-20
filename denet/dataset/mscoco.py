import os
import sys
import time
import numpy
import math
import json
import random

import denet.common as common
import denet.common.logging as logging
from denet.dataset import DatasetAbstract
from denet.dataset.image_loader import ImageLoader

class DatasetMSCOCO(DatasetAbstract):

    def get_data_shape(self):
        return (3, self.output_size, self.output_size)

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

        self.data=[]
        self.thread_num = thread_num

        param_str = ",".join(data_format.split(",")[1:])
        format_params = common.get_params_dict(param_str)
        self.data_types=[]
        if format_params.get("2014-train", False):
            self.data_types.append("train2014")
        if format_params.get("2014-val", False):
            self.data_types.append("val2014")
        if format_params.get("2014-test", False):
            self.data_types.append("test2014")
        if format_params.get("2015-test", False):
            self.data_types.append("test2015")
        if format_params.get("2015-test-dev", False):
            self.data_types.append("test-dev2015")

        if len(self.data_types) == 0:
            raise Exception("please specify mscoco subset")

        bbox_hist=[0 for _ in range(32)]

        self.images=[]
        self.class_labels = {}
        self.categories = None
        for data_type in self.data_types:

            if "test" in data_type:
                fname = os.path.join(input_dir, "annotations/image_info_%s.json"%data_type)
            else:
                fname = os.path.join(input_dir, "annotations/instances_%s.json"%data_type)

            json_data = common.json_from_file(fname)

            #get class labels
            data_categories={}
            for i, json_cat in enumerate(json_data["categories"]):
                data_categories[json_cat["id"]] = json_cat["name"]
                
                if not json_cat["name"] in self.class_labels:
                    self.class_labels[json_cat["name"]] = len(self.class_labels)

            assert (self.categories is None) or (self.categories == data_categories)
            self.categories=data_categories

            logging.verbose("Found %i labels:"%len(self.class_labels))

            #collect bounding boxes
            bboxs={}
            for json_ann in json_data.get("annotations", []):
                cls_id = self.class_labels[self.categories[json_ann["category_id"]]]
                image_id = json_ann["image_id"]
                bbox = json_ann["bbox"]
                    
                if not image_id in bboxs:
                    bboxs[image_id] = []
                bboxs[image_id].append((cls_id, (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])))

            logging.verbose("Found %i bboxs"%(sum([len(bbox) for bbox in bboxs.values()])))

            #collect images
            if data_type == "test-dev2015":
                data_type = "test2015"

            for image in json_data["images"]:
                fname = image["file_name"]
                image_id = image["id"]
                bbox_list = bboxs.get(image_id, [])
                bbox_hist[min(len(bbox_list), 31)] += 1
                self.images.append({"fname":os.path.join(input_dir, data_type, fname), "bboxs":bbox_list, "id":image_id})

        print("BBox histogram (%):", [round(100.0*x/sum(bbox_hist),1) for x in bbox_hist])
        
        #setup image loader
        self.image_loader = ImageLoader(thread_num, is_training, format_params)

        #subset
        self.output_size = self.image_loader.crop
        self.images_per_subset = format_params.get("images_per_subset", 10000)
        self.subset_total_size = len(self.images)
        self.subset_num = format_params.get("subset_num", sys.maxsize)
        self.subset_num = min(self.subset_num, int(math.ceil(self.subset_total_size / self.images_per_subset)))
        self.subset_index = -1
        self.subset_size = self.images_per_subset

        self.bbox_only = format_params.get("bbox_only", False)

        #only use samples with bounding boxes
        if self.image_loader.is_training and self.bbox_only:
            images_bbox=[]
            for image in self.images:
                if len(image["bboxs"]) > 0:
                    images_bbox.append(image)

            logging.info("Removed %i images without bboxs"%(len(self.images) - len(images_bbox)))
            self.images = images_bbox

        logging.info("Using MSCOCO dataset - size:", self.subset_total_size, "subset_num", self.subset_num, "images per subset:", self.subset_size, self.image_loader)

    #export detection in MSCOCO format
    def export_detections(self, output_fname, detection_list):
        

        label_cat_map = {}
        for index,name in self.categories.items():
            label_cat_map[self.class_labels[name]] = index

        results=[]
        for d in detection_list:
            image_id = d["meta"]["image"]["id"]
            scale_x, scale_y = d["meta"]["scale"]
            offset_x, offset_y = d["meta"]["offset"]
            image_width, image_height = d["meta"]["image_size"]
            dets = d["detections"]
            
            #export results
            dets.sort(key=lambda t:-t[0])
            for pr,cls,bbox in dets:
                x0 = max(min((bbox[0]*self.output_size + offset_x) / scale_x + 1, image_width), 1)
                y0 = max(min((bbox[1]*self.output_size + offset_y) / scale_y + 1, image_height), 1)
                x1 = max(min((bbox[2]*self.output_size + offset_x) / scale_x + 1, image_width), 1)
                y1 = max(min((bbox[3]*self.output_size + offset_y) / scale_y + 1, image_height), 1)

                results.append({"image_id": image_id, 
                                "category_id": label_cat_map[cls], 
                                "bbox": [round(x0,1), round(y0,1), round(x1-x0,1), round(y1-y0,1)], 
                                "score": round(pr,6)})

        with open(output_fname, "w") as f:
            json.dump(results, f)
