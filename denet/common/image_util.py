import os
import math
import numpy
from PIL import Image
import colorsys

def get_unique_colors(n, sat=1, val=1):
    hsv = [(x*1.0/n, sat, val) for x in range(n)]
    rgb = [colorsys.hsv_to_rgb(*x) for x in hsv]
    return rgb

#treat each value as a luminance between 0-1
def export_luminance(fname, data):
    assert data.ndim == 2

    d = numpy.maximum(0, numpy.minimum(255*data, 255))
    im_d = d.astype(numpy.uint8)
    # im_d = numpy.swapaxes(im_d, 0, 1)
    im = Image.fromarray(im_d, 'L')

    dname = os.path.dirname(fname)
    if dname != "" and not os.path.isdir(dname):
        os.makedirs(dname)

    im.save(fname)

#BBOX are in image normalized coordinates (xmin,ymin,xmax,ymax)
def export_activation_rgb(fname, data, bboxs=[]):

    if len(bboxs) > 0:
        data = numpy.copy(data)
        for x0,y0,x1,y1 in bboxs:
            x0 = max(0, min(int(x0*data.shape[2]), data.shape[2]-1))
            x1 = max(0, min(int(x1*data.shape[2]), data.shape[2]-1))
            y0 = max(0, min(int(y0*data.shape[1]), data.shape[1]-1))
            y1 = max(0, min(int(y1*data.shape[1]), data.shape[1]-1))

            data[:, y0, x0:x1] = 0
            data[:, y1, x0:x1] = 0
            data[:, y0:y1, x0] = 0
            data[:, y0:y1, x1] = 0
            data[0, y0, x0:x1] = 1
            data[0, y1, x0:x1] = 1
            data[0, y0:y1, x0] = 1
            data[0, y0:y1, x1] = 1

    d = numpy.maximum(0, numpy.minimum(255*data, 255))
    im_d = d.astype(numpy.uint8)
    im_d = numpy.swapaxes(im_d, 0, 2)
    im_d = numpy.swapaxes(im_d, 0, 1)
    im = Image.fromarray(im_d, 'RGB')

    dname = os.path.dirname(fname)
    if dname != "" and not os.path.isdir(dname):
        os.makedirs(dname)

    im.save(fname)



def wavelength_to_rgb(w, gamma=0.8):

    r=g=b=0.0
    if (w >= 380) and (w < 440):
        r = -(w-440)/(440-380)
        b = 1.0
    elif (w >= 440) and (w < 490):
        g = (w-440)/(490-440)
        b = 1.0
    elif (w >= 490) and (w < 510):
        g = 1.0;
        b = -(w - 510) / (510 - 490)
    elif (w >= 510) and (w < 580):
        r = (w - 510) / (580 - 510);
        g = 1.0;
    elif (w >= 580) and (w<645):
        r = 1.0;
        g = -(w - 645) / (645 - 580);
    elif (w >= 645) and (w<781):
        r = 1.0;
        
    factor = 0.0
    if (w >= 380) and (w<420):
        factor = 0.3 + 0.7*(w-380) / (420 - 380);
    elif (w >= 420) and (w<701):
        factor = 1.0;
    elif (w >= 701) and (w<781):
        factor = 0.3 + 0.7*(780 - w) / (780 - 700);
            
    return numpy.array([math.pow(r*factor, gamma), math.pow(g*factor, gamma), math.pow(b*factor, gamma)], dtype=numpy.float32);

def convert_hyperspec_rgb(data, wavelens, normalize=False):
    assert len(wavelens) == data.shape[0]

    image = numpy.zeros(shape=(3, data.shape[1], data.shape[2]), dtype=numpy.float32)
    for c in range(len(wavelens)):
        rgb = wavelength_to_rgb(wavelens[c]) / len(wavelens)
        image += rgb[:,None,None]*data[None,c,:,:]

    if normalize:
        return (image - image.min()) / (image.max()- image.min())
    else:
        return image

def export_hyperspec_rgb(fname, data, wavelens, normalize=False):
    assert len(wavelens) == data.shape[0]
    image = convert_hyperspec_rgb(data, wavelens, normalize)
    image = image.swapaxes(0,2)
    image = image.swapaxes(0,1)
    image = numpy.maximum(0, numpy.minimum(255*image, 255))
    image = image.astype(numpy.uint8)
    im = Image.fromarray(image, 'RGB')
    im.save(fname)

#data is integers for each label
def export_label(fname, data, colors, background=None):

    # labels = list(set(data.flatten().tolist()))

    if type(colors) is int:
        colors = [(255*r, 255*g, 255*b, 125) for r,g,b in get_unique_colors(colors)] 
        # print(colors)

    image = numpy.zeros(shape=(3, data.shape[0], data.shape[1]), dtype=numpy.float32)
    image_alpha = numpy.zeros(shape=(data.shape[0], data.shape[1]), dtype=numpy.float32)
    for c in range(data.max()+1):
        color = numpy.array(colors[c], dtype=numpy.uint8)
        image += color[0:3,None,None]*(data[None,:,:] == c)
        image_alpha += (color[3,None,None] / 255.0)*(data == c) 

    if not background is None:
        image = 255*background[None,:,:]*(1.0 - image_alpha[None,:,:]) + image*image_alpha[None,:,:]

    image = numpy.maximum(0, numpy.minimum(image, 255))
    image = image.astype(numpy.uint8)
    image = image.swapaxes(0,2)
    image = image.swapaxes(0,1)
    im = Image.fromarray(image, 'RGB')
    im.save(fname)

def export_activation(fname, data, dmin=None, dmax=None, border=1):

    if len(data.shape) == 2:
        data = data[None, :, :]
        border = 0

    assert len(data.shape) == 3
    dmin = data.min() if dmin is None else dmin
    dmax = data.max() if dmax is None else dmax

    n = int(math.ceil(math.sqrt(data.shape[-3])))
    h = data.shape[1]
    w = data.shape[2]

    im_x = numpy.zeros(((h+border)*n, (w+border)*n, 3), dtype=numpy.uint8)
    for i in range(1,n):
        im_x[:,i*(h+border)-1,0] = 255
        im_x[i*(h+border)-1,:,0] = 255

    for f in range(data.shape[0]):
        d = 255 * (data[f,:,:] - dmin) / (dmax - dmin)
        d = numpy.maximum(0, numpy.minimum(d, 255))
        d = d.astype(numpy.uint8)

        yi = (f // n)*(h + border)
        xi = (f % n)*(w + border)
        im_x[yi:(yi + h), xi:(xi+w), :] = d[:,:,None]

    im = Image.fromarray(im_x, 'RGB')
    dname = os.path.dirname(fname)
    if dname != "" and not os.path.isdir(dname):
        os.makedirs(dname)
    im.save(fname)
