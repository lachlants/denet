import numpy
import random
import math
from PIL import Image

import denet.common as common

#output image array as (C, H, W)
def image_to_array(im):
    if im.mode != "RGB":
        im = im.convert('RGB')

    im_x = numpy.array(im, dtype=numpy.float32)/255.0
    im_x = numpy.swapaxes(im_x, 0, 2)
    im_x = numpy.swapaxes(im_x, 1, 2)
    return im_x

#scale smallest / largest side to size x size image.
#scaling with largest side adds black borders
def scale(im, size, scale_mode = "small", interp_mode = Image.ANTIALIAS):
    old_size = im.size
    if scale_mode == "warp":
        new_size = (size, size)
    elif scale_mode == "small":
        if im.size[0] < im.size[1]:
            new_size = (size, int(math.ceil(size*im.size[1]/im.size[0])))
        else:
            new_size = (int(math.ceil(size*im.size[0]/im.size[1])), size)
    elif scale_mode == "large":
        if im.size[0] > im.size[1]:
            new_size = (size, int(math.ceil(size*im.size[1]/im.size[0])))
        else:
            new_size = (int(math.ceil(size*im.size[0]/im.size[1])), size)
    else:
        raise Exception("Unknown scale mode")

    if im.size[0] > new_size[0] and im.size[1] > new_size[1]:
        s = max(new_size[0], new_size[1])
        im.thumbnail((s, s), interp_mode)

    if im.size != new_size:
        im_resize = im.resize(new_size, interp_mode)
    else:
        im_resize = im

    assert im_resize.size == new_size, "Scaling Error! " + str(im.size) + " != " + str(new_size)
    return im_resize, new_size[0]/old_size[0], new_size[1]/old_size[1]

#adds a black border such that im is at least (size, size)
#ERROR somethings broken!
def add_border(im, size):
    if im.size[0] < size or im.size[1] < size:
        # raise Exception("error in add border, do not use!")
        size_new = (max(im.size[0], size), max(im.size[1], size))
        x = int((size_new[0] - im.size[0]) // 2)
        y = int((size_new[1] - im.size[1]) // 2)
        im_border = Image.new("RGB", size_new)
        im_border.paste(im, box=(x, y, x + im.size[0], y + im.size[1]))
        return im_border.copy(), -x, -y
    else:
        return im, 0, 0

#extract central crop
def center_crop(im, size):

    #add black border
    imm, x, y = add_border(im, size)

    #crop
    dx = math.ceil((imm.size[0] - size)/2)
    dy = math.ceil((imm.size[1] - size)/2)
    return imm.crop((dx, dy, dx+size, dy+size)), x+dx, y+dy

#extract random crop
def random_crop(im, size):

    #add black border
    imm, x, y = add_border(im, size)

    #crop
    dx = random.randint(0, imm.size[0]-size)
    dy = random.randint(0, imm.size[1]-size)
    return imm.crop((dx, dy, dx+size, dy+size)), x+dx, y+dy

#extract central, top left, top right, bottom left and bottom right crop
def multi_crop(im, size):
    center, x, y = center_crop(im, size)
    top_left = im.crop((0, 0, size, size))
    top_right = im.crop((im.size[0] - size, 0, im.size[0], size))
    bottom_left = im.crop((0, im.size[1] - size, size, im.size[1]))
    bottom_right = im.crop((im.size[0] - size, im.size[1] - size, im.size[0], im.size[1]))
    offset_x = [x, 0, im.size[0] - size, 0, im.size[0] - size]
    offset_y = [y, 0, 0, im.size[1] - size,  im.size[1] - size]
    return [center, top_left, top_right, bottom_left, bottom_right], offset_x, offset_y

#get multi_crop and their mirrors
def multi_crop_mirror(im, size):
    im_list, offset_x, offset_y = multi_crop(im, size)
    im_list += [x.transpose(Image.FLIP_LEFT_RIGHT) for x in im_list]
    offset_x += offset_x
    offset_y += offset_y
    mirror = [False]*5 + [True]*5
    return im_list, offset_x, offset_y, mirror

#lenet style crop / scale augmentation
def lenet_crop(im, size, area_min=0.08, aspect_factor=3/4, max_trials = 10, scale_mode="small"):
    area = im.size[0]*im.size[1]
    for _ in range(max_trials):
        target_area = random.uniform(area_min, 1.0)*area
        aspect_ratio = random.uniform(aspect_factor, 1.0/aspect_factor)
        w = int(math.sqrt(target_area * aspect_ratio))
        h = int(math.sqrt(target_area / aspect_ratio))
        if random.random() < 0.5:
            w, h = h, w

        if w <= im.size[0] and h <= im.size[1]:
            scale_x = size / w
            scale_y = size / h
            bbox_x0 = random.randint(0, im.size[0] - w)
            bbox_y0 = random.randint(0, im.size[1] - h)
            bbox_x1 = bbox_x0 + w
            bbox_y1 = bbox_y0 + h
            im_test = im.crop((bbox_x0, bbox_y0, bbox_x1, bbox_y1))
            im_test = im_test.resize((size,size), Image.BICUBIC)
            offset_x = bbox_x0*scale_x
            offset_y = bbox_y0*scale_y
            return im_test, scale_x, scale_y, offset_x, offset_y

    #fall back
    print("warning: using lenet crop fallback")
    im, scale_x, scale_y = scale(im, size, scale_mode)
    im, offset_x, offset_y = center_crop(im, size)
    return im, scale_x, scale_y, offset_x, offset_y

#resnet style crop / scale augmentation
def resnet_crop(im, size):

    target_size = random.randint(256, 480)
    if im.size[0] < im.size[1]:
        scaled_size = (target_size, int(math.ceil(target_size*im.size[1]/im.size[0])))
    else:
        scaled_size = (int(math.ceil(target_size*im.size[0]/im.size[1])), target_size)

    scale_x = scaled_size[0] / im.size[0]
    scale_y = scaled_size[1] / im.size[1]

    #crop position in scaled image
    offset_x = random.randint(0, scaled_size[0] - size)
    offset_y = random.randint(0, scaled_size[1] - size)

    #crop position in unscaled image
    bbox_x0 = int(offset_x / scale_x)
    bbox_y0 = int(offset_y / scale_y)
    bbox_x1 = int((offset_x + size) / scale_x)
    bbox_y1 = int((offset_y + size) / scale_y)

    #apply crop and scale
    im = im.crop((bbox_x0, bbox_y0, bbox_x1, bbox_y1))
    im, _, _ = scale(im, im_crop, scale_mode = "warp")
    return im, scale_x, scale_y, offset_x, offset_y

#SSD style crop
def ssd_crop(im, size, bboxs):

    im_size = max(im.size[0], im.size[1])
    im_border, offset_x, offset_y = add_border(im, im_size)

    crops=[(0, 0, im_size, im_size)]
    for min_jaccard_overlap in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        for _ in range(50):
            #in normalized coordinates [0,1)
            s = random.uniform(0.3, 1.0)
            # aspect_min = max(0.5, pow(s, +2))
            # aspect_max = min(2.0, pow(s, -2))
            # aspect_ratio = random.uniform(aspect_min, aspect_max)
            aspect_ratio = 1

            #in unnormalized coordinates [0,im_size)
            w = int(s * im.size[0] * math.sqrt(aspect_ratio))
            h = int(s * im.size[1] / math.sqrt(aspect_ratio))
            x0 = random.randint(0, im.size[0] - w)
            y0 = random.randint(0, im.size[1] - h)
            x1 = x0 + w
            y1 = y0 + h

            #check to see if any object bbox satisfies min_jaccard_overlap
            sx = size / w
            sy = size / h

            sx = size / w
            sy = size / h
            ox = (offset_x + x0)*sx
            oy = (offset_y + y0)*sy

            valid=False
            for bbox in bboxs:
                min_x = (bbox[0]*sx - ox) / size
                min_y = (bbox[1]*sy - oy) / size
                max_x = (bbox[2]*sx - ox) / size
                max_y = (bbox[3]*sy - oy) / size
                if common.overlap_iou((min_x,min_y,max_x,max_y)) > min_jaccard_overlap:
                    valid=True
                    break

            if valid:
                crops.append((x0,y0,x1,y1))
                break

    x0, y0, x1, y1 = random.choice(crops)
    sx, sy = size / (x1-x0), size / (y1-y0)
    ox = (offset_x + x0)*sx
    oy = (offset_y + y0)*sy

    interp_mode = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS])
    im = im_border.crop((x0, y0, x1, y1))
    im, _, _ = scale(im, size, scale_mode = "warp", interp_mode=interp_mode)
    return im, sx, sy, ox, oy

#randomly crop around randomly selected bbox
def denet_crop(im, size, bboxs, area_min=0.08, aspect_factor=1, max_trials = 10, interp_mode = Image.ANTIALIAS):

    im_size = max(im.size[0], im.size[1])
    im_border, offset_x, offset_y = add_border(im, im_size)
 
    for _ in range(max_trials):
        target_area = random.uniform(area_min, 1.0)*im_size*im_size
        aspect_ratio = pow(aspect_factor, random.uniform(-1.0, 1.0))

        w = int(math.sqrt(target_area * aspect_ratio))
        h = int(math.sqrt(target_area / aspect_ratio))

        if w > im_size or h > im_size:
            continue

        x0 = random.randint(0, im_size - w)
        y0 = random.randint(0, im_size - h)
        x1 = x0 + w
        y1 = y0 + h

        sx = size / w
        sy = size / h
        ox = (offset_x + x0)*sx
        oy = (offset_y + y0)*sy

        #make sure at least one object is onscreen
        for bbox in bboxs:
            min_x = (bbox[0]*sx - ox) / size
            min_y = (bbox[1]*sy - oy) / size
            max_x = (bbox[2]*sx - ox) / size
            max_y = (bbox[3]*sy - oy) / size
            overlap = common.overlap_rel((min_x,min_y,max_x,max_y))
            if overlap >= 0.5:
                im = im_border.crop((x0, y0, x1, y1))
                im, _, _ = scale(im, size, scale_mode = "warp", interp_mode=interp_mode)
                return im, sx, sy, ox, oy

    #fall back
    im, scale_x, scale_y = scale(im_border, size, interp_mode=interp_mode)
    return im, scale_x, scale_y, offset_x*scale_x, offset_y*scale_y

#random brightness / contrast / saturation augmentation on ndimage
def photometric(im_x, v = 0.4):
    assert type(im_x) is numpy.ndarray
    for op in random.sample(["contrast", "brightness", "saturation"], 3):
        alpha = random.uniform(1.0 - v, 1.0 + v)
        if op == "brightness":
            im_x = im_x*alpha
        elif op == "contrast":
            im_grey = 0.299*im_x[0,:,:] + 0.587*im_x[1,:,:] + 0.114*im_x[2,:,:]
            im_x = im_x*alpha + (1.0 - alpha)*numpy.mean(im_grey)
        elif op == "saturation":
            im_grey = 0.299*im_x[0,:,:] + 0.587*im_x[1,:,:] + 0.114*im_x[2,:,:]
            im_x = im_x*alpha + (1.0 - alpha)*im_grey[None,:,:]

    return im_x

#random colorspace distortion (Krizhevsky)
def colorspace(im_x, rgb_eigen_val, rgb_eigen_vec, v = 0.1):
    assert type(im_x) is numpy.ndarray
    aug = numpy.random.normal(0, v, 3)*rgb_eigen_val
    noise = numpy.dot(rgb_eigen_vec, aug.T)
    im_x += noise[:,None,None]
    return im_x

#test augmentation
if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Test Augmentation')
    parser.add_argument("--inputs", nargs="+", type=str, required=True, help="Input images")
    parser.add_argument("--num", type=int, default=10, help="Number of tests to run")
    parser.add_argument("--size", type=int, default=224, help="Number of tests to run")
    parser.add_argument("--func", type=str, required=True, help="Name of test function")
    args = parser.parse_args()

    func_args={}
    func_array=False
    if args.func == "lenet":
        func = lenet_crop
        func_args["size"] = args.size
    elif args.func == "resnet":
        func = resnet_crop
        func_args["size"] = args.size
    elif args.func == "scale":
        func = scale
        func_args["size"] = args.size
    elif args.func == "random_crop":
        func = random_crop
        func_args["size"] = args.size
    elif args.func == "center_crop":
        func = center_crop
        func_args["size"] = args.size
    elif args.func == "photometric":
        func = photometric
        func_array = True

    for fname in args.inputs:
        print("Loading " + fname)
        im = Image.open(fname)
        output = os.path.basename(fname)
        output = os.path.splitext(output)[0]
        im.save(output + "_original.png")

        if func_array:
            im = image_to_array(im)

        for i in range(args.num):
            im_r = func(im, **func_args)[0]
            if type(im_r) is list:
                for j,r in enumerate(im_r):
                    r.save(output + "_%s_%03i_%02i.png"%(args.func, i,j))
            else:
                im_r.save(output + "_%s_%03i.png"%(args.func, i))





