import os
import sys
import gzip
import time
import numpy
import pickle
import argparse
import math
import random
import theano

import denet.common as common
import denet.common.logging as logging
import denet.model.model_cnn as model_cnn
import denet.dataset as dataset

#center crop testing
def test_single(mode, model, data):

    y=[]
    yy=[]
    yt=[]
    for subset in range(data.subset_num):

        #load / prepare data
        logging.info("Subset %i: loading data..."%subset)
        data.load_from_subset(subset)
        labels = data.get_labels()

        #
        logging.info("Subset %i: computing error..."%subset)
        pr = model.predict_probabilities(data)
        for i in range(pr.shape[0]):
            pr_i = pr[i, :]
            top1 = numpy.argmax(pr_i)
            top5 = numpy.argpartition(-pr_i, 5)[:5]
            y.append(top1)
            yy.append(top5)
            yt.append(labels[i])

        logging.verbose("Samples - (True, Predicted):", list(zip(yt[-10:], y[-10:])))

    y = numpy.array(y, numpy.int64)
    yy = numpy.array(yy, numpy.int64)
    yt = numpy.array(yt, numpy.int64)

    error1 = numpy.sum(yt != y) / yt.shape[0]
    error5 = 1.0 - numpy.sum(numpy.any(yy[:,] == yt[:,None], axis=1)) / yt.shape[0]
    logging.info("Top1 - Error Rate: %.3f%%"%(100.0*error1))
    logging.info("Top5 - Error Rate: %.3f%%"%(100.0*error5))

#10 crop testing
def test_multicrop(mode, model, data):

    y=[]
    yy=[]
    yt=[]
    for subset in range(data.subset_num):

        #load / prepare data
        logging.info("Subset %i: loading data..."%subset)
        data.load_from_subset(subset)
        labels = data.get_labels()

        #
        logging.info("Subset %i: computing error..."%subset)
        pr = model.predict_probabilities(data)
        n = pr.shape[0] // 10
        for i in range(n):
            pr_i = numpy.sum(pr[i*10:(i+1)*10, :], axis=0)
            top1 = numpy.argmax(pr_i)
            top5 = numpy.argpartition(-pr_i, 5)[:5]
            y.append(top1)
            yy.append(top5)
            yt.append(labels[i*10])

    y = numpy.array(y, numpy.int64)
    yy = numpy.array(yy, numpy.int64)
    yt = numpy.array(yt, numpy.int64)

    error1 = 1.0 - numpy.sum(yt == y) / yt.shape[0]
    error5 = 1.0 - numpy.sum(numpy.any(yy[:,] == yt[:,None], axis=1)) / yt.shape[0]
    logging.info("Top1 - Error Rate: %.3f%%"%(100.0*error1))
    logging.info("Top5 - Error Rate: %.3f%%"%(100.0*error5))


def export_detection_image(fname, data, class_labels_inv, dets=[]):

    bboxs = []
    for pr,cls,bbox in dets:
        x0,y0,x1,y1 = bbox
        x0 = max(0, min(int(x0*data.shape[2]), data.shape[2]-1))
        x1 = max(0, min(int(x1*data.shape[2]), data.shape[2]-1))
        y0 = max(0, min(int(y0*data.shape[1]), data.shape[1]-1))
        y1 = max(0, min(int(y1*data.shape[1]), data.shape[1]-1))
        bboxs.append((cls, x0, y0, x1, y1))

    bboxs.sort(key=lambda t:(t[3]-t[1])*(t[4]-t[2]))
    bboxs = bboxs[::-1]

    #create unique color map for 
    import colorsys
    cls_unique = set([cls for cls,_,_,_,_ in bboxs])
    cls_unique_n = len(cls_unique)
    colormap_hsv = [(x*1.0/cls_unique_n, 0.5, 0.5) for x in range(cls_unique_n)]
    colormap_rgb = map(lambda x: colorsys.hsv_to_rgb(*x), colormap_hsv)
    cls_colormap = {cls:rgb for cls,rgb in zip(cls_unique,colormap_rgb)}

    #construct data filter
    bbox_alpha = 0.75
    data_filter = numpy.zeros(shape=(4, data.shape[1], data.shape[2]), dtype=numpy.float32)
    for cls,x0,y0,x1,y1 in bboxs:
        rgb = cls_colormap[cls]
        rgba_border = numpy.array([rgb[0], rgb[1], rgb[2], 1.0], dtype=numpy.float32)
        rgba_center = numpy.array([rgb[0], rgb[1], rgb[2], bbox_alpha], dtype=numpy.float32)
        data_filter[:, y0:y1, x0:x1] = rgba_center[:,None,None]
        for i in range(2):
            if (y0-i) > 0:
                data_filter[:, y0-i, x0:x1] = rgba_border[:,None]
            if (y1+i) < data.shape[1]:
                data_filter[:, y1+i, x0:x1] = rgba_border[:,None]
            if (x0-i) > 0:
                data_filter[:, y0:y1, x0-i] = rgba_border[:,None]
            if (x1+i) < data.shape[2]:
                data_filter[:, y0:y1, x1+i] = rgba_border[:,None]

    #apply filter
    data = data_filter[3,None,:,:]*data_filter[:3,:,:] + (1.0 - data_filter[3,None,:,:])*data[:,:,:]
    d = numpy.maximum(0, numpy.minimum(255*data, 255))
    im_d = d.astype(numpy.uint8)
    im_d = numpy.swapaxes(im_d, 0, 2)
    im_d = numpy.swapaxes(im_d, 0, 1)

    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont
    im = Image.fromarray(im_d, 'RGB')

    #add legend:
    im_legend = Image.new("RGB", (im.size[0] + 256, im.size[1]), "black")
    im_legend.paste(im, (0,0,im.size[0],im.size[1]))

    draw = ImageDraw.Draw(im_legend)
    font_height = 16
    # font = ImageFont.truetype("sans-serif.ttf", font_height)
    font = ImageFont.truetype("DejaVuSerif-Bold.ttf", font_height)
    # font = ImageFont.load_default()
    for i,item in enumerate(cls_colormap.items()):
        cls,rgb = item
        rgb = (int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255))
        rgb_fill = (int(rgb[0]*bbox_alpha),int(rgb[1]*bbox_alpha),int(rgb[2]*bbox_alpha))
        x0 = im.size[0]
        y0 = i*(font_height+4)

        draw.rectangle((x0,y0,x0+16,y0+16), fill=rgb_fill, outline=rgb)
        draw.text((x0+18, y0), class_labels_inv[cls][0].upper() + class_labels_inv[cls][1:], rgb, font=font)

    #save image
    dname = os.path.dirname(fname)
    if dname != "" and not os.path.isdir(dname):
        os.makedirs(dname)

    im_legend.save(fname)


def test_detector(mode, model, data, output_fname, params):

    detect_params = common.get_params_dict(params)
    print("Using detector params:", detect_params, params)

    detect_layer = model.layers[-1]
    class_labels_inv = {v:k for k,v in model.class_labels.items()}
    index=0
    detections=[]
    for subset in range(data.subset_num):

        #load data
        logging.info("Subset %i: loading data..."%subset)
        data.load_from_subset(subset)
        data_x, data_m, data_size = data.export(model.batch_size)

        #
        logging.info("Subset %i: computing error..."%subset)
        batch_num = data_x.shape[0] // model.batch_size
        subset_det=[]
        for n in range(batch_num):
            dx = data_x[n*model.batch_size:(n+1)*model.batch_size]
            dm = data_m[n*model.batch_size:(n+1)*model.batch_size]
            results = detect_layer.get_detections(model, dx, dm, detect_params)

            #export samples images
            if "image" in mode:
                for i, r in enumerate(results):
                    export_detection_image("%06i_dets.png"%(index+i), dx[i,:,:,:], class_labels_inv, r["detections"])

                    # for cls in set(r["meta"]["class"]):
                    #     objs = []
                    #     for obj_cls,obj in zip(r["meta"]["class"], r["meta"]["bbox"]):
                    #         if obj_cls == cls:
                    #             objs.append(obj)
                    #     common.export_activation_rgb("%06i_gt_%s.png"%(index+i, class_labels_inv[cls]), data_x[i,:,:,:], objs)

                    # for j, det in enumerate(r["detections"]):
                    #     pr, cls, bbox = det
                    #     logging.verbose("sample %i - %i: class %s, pr=%.4f, bbox:"%(index+i, j, class_labels_inv[cls], pr), bbox)
                    #     common.export_activation_rgb("%06i_%03i_%s_pr%0.2f.png"%(index+i, j, class_labels_inv[cls], pr), data_x[i,:,:,:], [bbox])

            subset_det += results
            index += model.batch_size

        detections += subset_det[:data_size]


    logging.info("Found %i detections for %i samples"%(sum([len(d["detections"]) for d in detections]), len(detections)))

    #export raw
    fname = os.path.join(os.path.dirname(output_fname), "detections.json")
    logging.info("Saving raw detections to:" + fname)
    json_dets={"dets":detections, "classLabels":model.class_labels, "detectParams":detect_params}
    common.json_to_file(fname, json_dets)

    if "voc" in mode:
        from denet.dataset.pascal_voc import DatasetPascalVOC
        logging.info("Exporting pascal voc detections to: ", os.path.dirname(output_fname))
        _, _, height, width = model.get_input_shape()
        DatasetPascalVOC.export_detections(os.path.dirname(output_fname), detections, width, height, class_labels_inv)
        DatasetPascalVOC.get_precision(detections, detect_params.get("matchIOU", 0.5))

    elif "mscoco" in mode:
        from denet.dataset.mscoco import DatasetMSCOCO
        logging.info("Exporting mscoco detections to: ", output_fname + ".json")
        data.export_detections(output_fname + ".json", detections)

    elif "imagenet" in mode:
        from denet.dataset.imagenet import DatasetImagenet
        DatasetImagenet.get_localization_error(detections)

def test_segment(mode, model, data):

    cls_tp = [0 for _ in range(model.class_num)]
    cls_tn = [0 for _ in range(model.class_num)]
    cls_fp = [0 for _ in range(model.class_num)]
    cls_fn = [0 for _ in range(model.class_num)]
    for subset in range(data.subset_num):

        #load data
        logging.info("Subset %i: loading data..."%subset)
        data.load_from_subset(subset)

        #predict output
        pr = model.predict_output(data)
        label = numpy.argmax(pr, axis=1)
        for b in range(pr.shape[0]):

            label_predict = label[b,:,:]
            label_true = data.data[b][2]["label"]

            for cls in range(model.class_num):
                tp = numpy.logical_and(label_true == cls, label_predict == cls)
                fn = numpy.logical_and(label_true == cls, label_predict != cls)
                fp = numpy.logical_and(label_true != cls, label_predict == cls)
                tn = numpy.logical_and(label_true != cls, label_predict != cls)
                cls_tp[cls] += numpy.count_nonzero(tp)
                cls_fn[cls] += numpy.count_nonzero(fn)
                cls_fp[cls] += numpy.count_nonzero(fp)
                cls_tn[cls] += numpy.count_nonzero(tn)

            fname = data.data[b][0]
            hyperspec = data.data[b][1]
            hyperspec_gray = hyperspec.mean(axis=0)
            
            colors=[(255,255,255,0), (255,0,0,60), (0,0,255,60)]
            common.export_label(fname + "_predict.png", label_predict, colors, background=hyperspec_gray)
            common.export_label(fname + "_true.png", label_true, colors, background=hyperspec_gray)

    class_labels_inv = {v:k for k,v in model.class_labels.items()}
    overall_correct=0
    overall_total=0
    for cls in range(model.class_num):
        
        correct = cls_tp[cls] + cls_tn[cls]
        total = cls_tp[cls] + cls_fp[cls] + cls_fn[cls] + cls_tn[cls]
        overall_correct += correct
        overall_total += total
        t=(class_labels_inv[cls], 100*correct/total, 100*cls_tp[cls] / (cls_tp[cls] + cls_fn[cls]), 100*cls_fp[cls] / (cls_fp[cls] + cls_tn[cls]))
        print("Accuracy (%s): %.2f%%, TPR: %.2f%%, FPR: %.2f%% "%t)
        

    print("Overall Accuracy: %.2f%%"%(100*overall_correct / overall_total))


def main():

    #load arguments:
    parser = argparse.ArgumentParser(description='Predict labels using trained model')
    logging.add_arguments(parser)
    parser.add_argument("--model", required=True, help="the model file")
    parser.add_argument("--input", required=True, help="The folder with data")
    parser.add_argument("--results", default="./results", type=str, help="Results folder / filename")
    parser.add_argument("--extension", default="png", help="Image file extension")
    parser.add_argument("--batch-size", type=int, default=100, help="Size of processing batchs")
    parser.add_argument("--predict-mode", default="single", required=False, help="Prediction mode (single, multicrop, convolutional)")
    parser.add_argument("--thread-num", default=1, type=int, help="Number of threads for dataset loading")
    parser.add_argument("--params", default="", type=str, help="Additional detection params")
    args = parser.parse_args()

    logging.init(args)

    logging.info("------------------------")
    model = model_cnn.load_from_file(args.model, args.batch_size)
    logging.info("Class labels:\n", model.class_labels)

    logging.info("------------------------")
    logging.info("Loading Dataset...")
    data = dataset.load(args.input, args.extension, class_labels=model.class_labels, thread_num=args.thread_num)

    logging.info("------------------------")
    logging.info("Using cudnn algo_fwd: ", theano.config.dnn.conv.algo_fwd)
    logging.info("------------------------")

    if "single" in args.predict_mode:
        logging.info("Testing: single crop")
        test_single(args.predict_mode, model, data)

    elif "multicrop" in args.predict_mode:
        logging.info("Testing: multicrop")
        assert "multicrop" in args.extension
        test_multicrop(args.predict_mode, model, data)

    elif "segment" in args.predict_mode:
        logging.info("Testing: segmentation")
        test_segment(args.predict_mode, model, data)

    elif "detect" in args.predict_mode:
        logging.info("Testing: Detection")
        test_detector(args.predict_mode, model, data, args.results, args.params)

    logging.info("Done")

if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    sys.exit(main())
