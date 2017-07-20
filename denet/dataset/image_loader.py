import os
from PIL import Image
import multiprocessing as mp
import numpy
import random

import denet.common as common
import denet.common.logging as logging
import denet.dataset.augment as augment

def load_sample_proc(args):

    image = args["image"]
    image_fname = image["fname"]
    image_bboxs = image.get("bboxs", [])
    bboxs = [bbox for _,bbox in image_bboxs]
    image_cls = image.get("class", None)

    is_training = args["isTraining"]
    scale = args["scale"]
    scale_mode = args.get("scaleMode", "small")
    crop = args["crop"]
    crop_mode = args.get("cropMode", "default")
    max_trials = args.get("maxTrials", 10)
    multicrop = args.get("multicrop", False)

    area_min = args.get("areaMin", 0.08)
    aspect_factor = args.get("aspectFactor", 3/4)

    check_onscreen = args.get("checkOnscreen", 0.0)
    check_center = args.get("checkCenter", False)

    augment_mirror = args.get("augmentMirror", False)
    augment_color = args.get("augmentColor", False)
    augment_photo = args.get("augmentPhoto", False)

    subtract_mean = args.get("subtractMean", False)
    if subtract_mean:
        rgb_mean = numpy.array(args["rgbMean"], dtype=numpy.float32)
        rgb_std = numpy.array(args["rgbStd"], dtype=numpy.float32)
    if augment_color:
        rgb_eigen_val = numpy.array(args["rgbEigenVal"], dtype=numpy.float32)
        rgb_eigen_vec = numpy.array(args["rgbEigenVec"], dtype=numpy.float32)

    seed = args.get("seed", None)
    random.seed(seed)
    numpy.random.seed(seed)

    im = Image.open(image_fname)
    im_size = im.size
    mirror = False
    if is_training:

        #scale / crop augmentation
        if crop_mode == "resnet":
            im, scale_x, scale_y, offset_x, offset_y = augment.resnet_crop(im, crop, scale, 480)
        elif crop_mode == "lenet":
            im, scale_x, scale_y, offset_x, offset_y = augment.lenet_crop(im, crop, area_min, aspect_factor, max_trials, scale_mode)
        elif crop_mode == "denet":
            im, scale_x, scale_y, offset_x, offset_y = augment.denet_crop(im, crop, bboxs, area_min, aspect_factor, max_trials)
        elif crop_mode == "ssd":
            im, scale_x, scale_y, offset_x, offset_y = augment.ssd_crop(im, crop, bboxs)
        elif crop_mode == "default":
            im, scale_x, scale_y = augment.scale(im, scale, scale_mode)
            im, offset_x, offset_y = augment.random_crop(im, crop)
        elif crop_mode == "center":
            im, scale_x, scale_y = augment.scale(im, scale, scale_mode)
            im, offset_x, offset_y = augment.center_crop(im, crop)
        else:
            raise Exception("Unknown crop mode:", crop_mode)

        #convert to ndarray
        im_x = augment.image_to_array(im)
        if augment_photo:
            im_x = augment.photometric(im_x)
        if augment_color:
            im_x = augment.colorspace(im_x, rgb_eigen_val, rgb_eigen_vec)

        #random flip
        if augment_mirror and random.random() >= 0.5:
            im_x = im_x[:,:,::-1]
            mirror = True

        im_list_x = [im_x]
    else:
        im, scale_x, scale_y = augment.scale(im, scale, scale_mode)
        if multicrop:
            im_list, offset_x, offset_y, mirror = augment.multi_crop_mirror(im, crop)
            im_list_x = [augment.image_to_array(x) for x in im_list]
            scale_x = [scale_x]*10
            scale_y = [scale_y]*10
        else:
            im, offset_x, offset_y = augment.center_crop(im, crop)
            im_list_x = [augment.image_to_array(im)]

    data=[]
    for i in range(len(im_list_x)):

        #collect info for sample
        im_x = im_list_x[i] if type(im_list_x) is list else im_list_x

        if subtract_mean:
            im_x = (im_x - rgb_mean[:,None,None]) / rgb_std[:,None,None]

        x = offset_x[i] if type(offset_x) is list else offset_x
        y = offset_y[i] if type(offset_y) is list else offset_y
        sx = scale_x[i] if type(scale_x) is list else scale_x
        sy = scale_y[i] if type(scale_y) is list else scale_y
        m = mirror[i] if type(mirror) is list else mirror

        #collect meta data
        bboxs = []
        classes = []
        for cls, bbox in image_bboxs:
            min_x = (bbox[0]*sx - x) / crop
            min_y = (bbox[1]*sy - y) / crop
            max_x = (bbox[2]*sx - x) / crop
            max_y = (bbox[3]*sy - y) / crop
            if m:
                x0,x1 = min_x, max_x
                max_x = 1.0 - x0
                min_x = 1.0 - x1

            cx = (min_x + max_x)*0.5
            cy = (min_y + max_y)*0.5
            bbox = (common.clip(min_x,0,1), common.clip(min_y,0,1), common.clip(max_x,0,1), common.clip(max_y,0,1))
            if common.overlap_rel((min_x,min_y,max_x,max_y)) >= check_onscreen:
                bboxs.append(bbox)
                classes.append(cls)
            elif check_center and cx >= 0.0 and cx <= 1.0 and cy >= 0.0 and cy <= 1.0:
                bboxs.append(bbox)
                classes.append(cls)

        meta = {"class":classes, "bbox":bboxs, "scale":(sx, sy),"offset":(x,y), "mirror":m, "image_size":im_size, "image": image}
        if not image_cls is None:
            meta["image_class"] = image_cls

        data.append((os.path.basename(image_fname), im_x, meta))

    return data

class ImageLoader:
    def __init__(self, thread_num, is_training, format_params={}):
        #cropping / scaling
        self.crop = format_params.get("crop", 224)
        self.multicrop = format_params.get("multicrop", False)
        self.crop_mode = format_params.get("crop_mode", "default")
        self.max_trials = format_params.get("max_trials", 10)
        self.scale = format_params.get("scale", self.crop)
        self.scale_mode = format_params.get("scale_mode", "small")
        self.area_min = format_params.get("area_min", 0.08)
        self.aspect_factor = format_params.get("aspect_factor", 0.75)

        #other augmentation
        self.subtract_mean = format_params.get("subtract_mean", False)
        self.augment_color = format_params.get("augment_color", False)
        self.augment_photo = format_params.get("augment_photo", False)
        self.check_onscreen = format_params.get("check_onscreen", 0.5)
        self.check_center = format_params.get("check_center", False)
        self.augment_mirror = True
        self.rgb_mean = numpy.zeros(3, dtype=numpy.float32)
        self.rgb_std = numpy.zeros(3, dtype=numpy.float32)
        self.rgb_eigen_val = numpy.zeros(3, dtype=numpy.float32)
        self.rgb_eigen_vec = numpy.zeros((3,3), dtype=numpy.float32)

        self.is_training = is_training
        self.thread_num = thread_num
        self.procs = mp.Pool(self.thread_num)

    #print some useful stuff
    def __str__(self):
        r = "thread_num: %i, is_training: %i, subtract_mean: %i, scale: %i, scale mode: %s, "%(self.thread_num, self.is_training, self.subtract_mean, self.scale, self.scale_mode)
        r += "crop: %i, crop_mode: %s, multicrop: %i, onscreen: %.1f, center: %i, "%(self.crop, self.crop_mode, self.multicrop, self.check_onscreen, self.check_center)
        r += "area: (%.2f,1.0), aspect: (%.2f,%.2f), max_trials: %i, "%(self.area_min, self.aspect_factor, 1.0/self.aspect_factor, self.max_trials)
        r += "augment - mirror: %i, color: %i, photo: %i"%(self.augment_mirror, self.augment_color, self.augment_photo)
        return r

    def load(self, images):

        #build args
        args_list=[]
        for image in images:
            args = {"image": image,
                    "isTraining" : self.is_training,
                    "multicrop" : self.multicrop,
                    "checkOnscreen" : self.check_onscreen,
                    "checkCenter" : self.check_center,
                    "scale" : self.scale,
                    "scaleMode" : self.scale_mode,
                    "crop" : self.crop,
                    "cropMode" : self.crop_mode,
                    "subtractMean": self.subtract_mean,
                    "maxTrials": self.max_trials,
                    "areaMin": self.area_min,
                    "aspectFactor": self.aspect_factor,
                    "rgbMean" : self.rgb_mean.tolist(),
                    "rgbStd" : self.rgb_std.tolist(),
                    "rgbEigenVec" : self.rgb_eigen_vec.tolist(),
                    "rgbEigenVal" : self.rgb_eigen_val.tolist(),
                    "augmentMirror" : self.augment_mirror,
                    "augmentColor" : self.augment_color,
                    "augmentPhoto" : self.augment_photo,
                    "seed": random.randint(0,1000000)}

            args_list.append(args)

        data_list = self.procs.imap(load_sample_proc, args_list)
        data = sum(data_list, [])
        return data


#TESTING
if __name__ == '__main__':
    import argparse
    import os
    import dataset
    import math

    parser = argparse.ArgumentParser(description='Test Image Loader')
    parser.add_argument("--input", type=str, required=True, help="Input images")
    parser.add_argument("--extension", default="")
    parser.add_argument("--thread-num", type=int, default=1, help="")
    parser.add_argument("--index", type=int, default=0, help="sample index")
    parser.add_argument("--num", type=int, default=10, help="Number of tests to run")
    parser.add_argument("--training", default=False, action="store_true")
    args = parser.parse_args()

    dataset = dataset.load(args.input, args.extension, args.thread_num)
    images = [dataset.images[args.index]]*args.num
    # images = dataset.images
    print("Image:", images[0])

    format_params={}
    for str in args.extension.split(",")[1:]:
        pv = str.split("=")
        param = pv[0]
        value = common.convert_num(pv[1]) if len(pv) > 1 else True
        format_params[param] = value

    image_loader = ImageLoader(args.thread_num, args.training, format_params)
    data = image_loader.load(images)
    print("Image Loader:", image_loader)

    index=0
    for _, im_x, meta in data:
        print("Sample %i: "%index, meta)
        common.export_activation_rgb("%06i.png"%index, im_x, meta["bbox"])
        index += 1

    #bbox stats
    # scale=numpy.zeros(10)
    # aspect=numpy.zeros(10)
    # num=0
    # for _,_,meta in data:
        # for bbox in meta["objs"]:
            # w = bbox[2]-bbox[0]
            # h = bbox[3]-bbox[1]
            # s = max(w, h)
            # a = h/w
            # print("scale, aspect:", (s, a))

            # scale[max(0, min(math.floor(s*scale.shape[0]), scale.shape[0]-1))] += 1
            # aspect[max(0, min(math.floor(a*10), 10-1))] += 1
            # num += 1

    # print("scale dist: ", (100.0 * scale / num).tolist())
    # print("aspect dist: ", (100.0 * aspect / num).tolist())

    print("Done")
