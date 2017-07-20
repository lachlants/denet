#include <Python.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "numpy/arrayobject.h"
#include "theano_mod_helper.h"

typedef std::tuple<float, float, float, float, float, npy_intp, npy_intp> Instance;
static float get_overlap_iou(const Instance& inst_a, const Instance& inst_b){
    float x0_a = std::get<1>(inst_a);
    float y0_a = std::get<2>(inst_a);
    float x1_a = std::get<3>(inst_a);
    float y1_a = std::get<4>(inst_a);

    float x0_b = std::get<1>(inst_b);
    float y0_b = std::get<2>(inst_b);
    float x1_b = std::get<3>(inst_b);
    float y1_b = std::get<4>(inst_b);

    float dx = std::max(0.0f, std::min(x1_a, x1_b) - std::max(x0_a, x0_b));
    float dy = std::max(0.0f, std::min(y1_a, y1_b) - std::max(y0_a, y0_b));
    float area_intersect = dx*dy;
    float area_a = (x1_a - x0_a)*(y1_a - y0_a);
    float area_b = (x1_b - x0_b)*(y1_b - y0_b);
    float area_union = area_a + area_b - area_intersect;
    return area_intersect / area_union;
};

static PyObject* build_detections_nms(PyObject *self, PyObject *args) {

    //get args
    float pr_threshold; 
    float nms_threshold; 
    PyObject* det_obj; //ndarray
    PyObject* bbox_obj; //ndarray
    PyObject* bbox_num; //list of int
    if (!PyArg_ParseTuple(args, "ffOOO", &pr_threshold, &nms_threshold, &det_obj, &bbox_obj, &bbox_num))
        return NULL;

    PyArrayObject* det_pr = (PyArrayObject*)PyArray_FROM_OTF(det_obj, NPY_FLOAT, NPY_IN_ARRAY);
    if (det_pr == NULL)
        return NULL;

    PyArrayObject* bbox = (PyArrayObject*)PyArray_FROM_OTF(bbox_obj, NPY_FLOAT, NPY_IN_ARRAY);
    if (bbox == NULL)
        return NULL;

    float log_pr_threshold = std::log(pr_threshold);
    npy_intp batch_size = PyArray_DIM(det_pr, 0);
    npy_intp class_num = PyArray_DIM(det_pr, 1) - 1;
    npy_intp sample_num = PyArray_DIM(det_pr, 2);

    PyObject* det_lists = PyList_New(0);
    for(npy_intp b=0; b < batch_size; b++){

        long batch_bbox_num = PyLong_AS_LONG(PyList_GET_ITEM(bbox_num, b));
        PyObject* batch_dets = PyList_New(0);
        size_t before_nms=0, after_nms=0;
        for(npy_intp cls=0; cls < class_num; cls++){

            std::vector<Instance> instances;
            for(npy_intp j=0; (j < sample_num) && ((j*sample_num) < batch_bbox_num); j++){
                for(npy_intp i=0; (i < sample_num) && ((j*sample_num + i) < batch_bbox_num); i++){

                    float log_pr = *(float*)PyArray_GETPTR4(det_pr, b, cls, j, i);
                    if (log_pr >= log_pr_threshold){
                        float* det_bbox = (float*)PyArray_GETPTR4(bbox, b, j, i, 0);
                        instances.push_back(Instance(log_pr, det_bbox[0], det_bbox[1], det_bbox[2], det_bbox[3], j, i));
                    }
                }
            }
            before_nms += instances.size();

            //run NMS
            std::vector<Instance> instances_nms;
            if ((nms_threshold > 0.0) && (nms_threshold < 1.0)){
                
                for(const Instance& inst_a: instances){
                    bool unique=true;
                    for(const Instance& inst_b : instances){
                        if ((std::get<0>(inst_a) < std::get<0>(inst_b)) && (get_overlap_iou(inst_a, inst_b) > nms_threshold)){
                            unique = false;
                            break;
                        }
                    }
                    if (unique)
                        instances_nms.push_back(inst_a);
                };
                
            } else {
                instances_nms = instances;
            }
            after_nms += instances_nms.size();

            for(const Instance& inst: instances_nms){
                PyObject* det = Py_BuildValue("fi(ffff)", std::exp(std::get<0>(inst)), cls, std::get<1>(inst), std::get<2>(inst), std::get<3>(inst), std::get<4>(inst));
                PyList_Append(batch_dets, det);
                Py_DECREF(det);
            }
        }
        // printf("%i - NMS before: %zu, after %zu\n", b, before_nms, after_nms);
        PyList_Append(det_lists, batch_dets);
        Py_DECREF(batch_dets);
    }
    
    Py_DECREF(det_pr);
    Py_DECREF(bbox);
    return det_lists;

}

static PyMethodDef methods[] = {
        {"build_detections_nms", build_detections_nms, METH_VARARGS, ""},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "denet_detect",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_denet_detect(void){
    import_array();
    PyObject *m = PyModule_Create(&moduledef);
    return m;
};
