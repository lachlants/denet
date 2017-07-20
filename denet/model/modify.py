import argparse
import json
import numpy
import gzip
import random

import denet.common as common
import denet.common.logging as logging
import denet.model.model_cnn as model_cnn

parser = argparse.ArgumentParser(description='Modify CNN model')
logging.add_arguments(parser)
parser.add_argument("--seed", type=int, default=23455, help="Random Seed for weights")
parser.add_argument("--input", type=str, help="")
parser.add_argument("--output", type=str, help="")
parser.add_argument("--class-num", type=int, default=None)
parser.add_argument("--image-size", nargs="+", type=int, default=None)
parser.add_argument("--use-cudnn-pool", default=False,action="store_true")
parser.add_argument("--optimize-bn", default=False, action="store_true", help="Freeze batch normalization layers")
parser.add_argument("--convert-bn-relu", default=False, action="store_true")
parser.add_argument("--merge", default=False, action="store_true", help="merge split layers")
parser.add_argument("--modify-bn", default=None, nargs="+", type=str, help="update momentum / eps for batch norm")
parser.add_argument("--modify-layer", default=None, nargs="+", type=str, help="modify layer")
parser.add_argument("--layer-insert", default=[], nargs="+", help="insert layer at position N:DESC")
parser.add_argument("--layer-remove", default=0, type=int, help="remove N layer from end")
parser.add_argument("--layer-append", default=[], nargs="+", type=str, help="append layers to end")
parser.add_argument("--border-mode", default="half", help="Border mode for convolutional layers (full, valid)")
parser.add_argument("--activation", default="relu", help="Activation function used in convolution / hidden layers (tanh, relu, leaky-relu)")
parser.add_argument("--weight-init", nargs="+", default=["he-backward"], help="Weight initialization scheme")
args = parser.parse_args()

#init 
logging.init(args)
random.seed(args.seed)
numpy.random.seed(args.seed)

model = model_cnn.load_from_file(args.input)

model_reload = False
if not args.class_num is None:
    model.class_num = args.class_num
    model_reload = True

if not args.image_size is None:
    model.data_shape = (3, args.image_size[1], args.image_size[0])
    model_reload = True

for layer in model.layers:
    if (layer.type_name == "activation" or layer.type_name == "resnet") and layer.activation != args.activation:
        layer.activation = args.activation
        model_reload = True

if args.merge:
    print("Merging split layers")
    model_reload = True
    for layer in model.layers:
        if layer.type_name == "split":
            layer.enabled = False
        elif layer.type_name == "skip-src":
            layer.has_split = False
    
if args.use_cudnn_pool:
    print("Modifying pool layer")
    for layer in model.layers:
        if layer.type_name == "pool" and not layer.ignore_border:
            layer.pad = (1,1)
            layer.ignore_border = True
            model_reload = True

if args.convert_bn_relu:
    from denet.layer import InitialLayer
    from denet.layer.batch_norm_relu import BatchNormReluLayer

    def convert_bn(layer):
        bn_json = layer.export_json()
        bn_relu = BatchNormReluLayer([InitialLayer(layer.input, layer.input_shape)], json_param=bn_json)
        bn_relu.import_json(bn_json)
        return bn_relu

    model_reload = True
    model_layers = [model.layers[0]]
    for i in range(1,len(model.layers)-1):

        if (model.layers[i].type_name, model.layers[i+1].type_name) == ("batchnorm","activation") and model.layers[i+1].activation == "relu":
            print("Merging batchnorm+relu layer")
            model_layers.append(convert_bn(model.layers[i]))

        elif (model.layers[i-1].type_name, model.layers[i].type_name) == ("batchnorm","activation") and model.layers[i].activation == "relu":
            pass

        elif model.layers[i].type_name == "resnet" and not ("bnrelu" in model.layers[i].version):

            print("Updating resnet layer")
            if "preactivation" in model.layers[i].version:
                pass
            else:
                model.layers[i].layers[2] = convert_bn(model.layers[i].layers[2])
                del model.layers[i].layers[3]

                if model.layers[i].bottleneck > 0:
                    model.layers[i].layers[4] = convert_bn(model.layers[i].layers[4])
                    del model.layers[i].layers[5]
                    
                    
            model.layers[i].version += ",bnrelu"
            model_layers.append(model.layers[i])

        else:
            model_layers.append(model.layers[i])
        
    model.layers = model_layers + [model.layers[-1]]

if not args.modify_bn is None:
    print("modifying batch normalization layers")
    
    json_update = {"enabled" : bool(args.modify_bn[0]), 
                   "momentum": float(args.modify_bn[1]), 
                   "eps": float(args.modify_bn[2]) }
    print("Updating batch norm layers:", json_update)
    for layer in model.layers:
        if layer.type_name == "batchnorm":
            layer.enabled = json_update["enabled"]
            layer.momentum = json_update["momentum"]
            layer.eps = json_update["eps"]
        elif layer.type_name == "resnet":
            layer.bn_json_param.update(json_update)

    model_reload = True

if not args.modify_layer is None:

    layer_name = args.modify_layer[0]
    layer_param = args.modify_layer[1:]
    for layer in model.layers:
        if layer.type_name == layer_name:
            for param in layer_param:
                ps = param.split("=")
                val_name = ps[0]
                val_type = type(getattr(layer, val_name))
                if val_type is bool:
                    val = {"True":True, "False":False, "0":False, "1":True}[ps[1]]
                else:
                    val = val_type(ps[1])
                print("%s - modifying param %s from"%(layer_name, val_name), getattr(layer, val_name), "to", val)
                setattr(layer, val_name, val)
            break

    model_reload = True
           
if args.layer_remove > 0:
    print("Removing Layers: ", [layer.type_name for layer in model.layers[-args.layer_remove:]])
    model.layers = model.layers[:-args.layer_remove]
    model_reload = True

#reload model 
if model_reload:
    model_cnn.save_to_file(model, args.output)
    model = model_cnn.load_from_file(args.output)
    model_reload = False

if len(args.layer_insert) > 0:
    print("Inserting new layers: ", args.layer_insert)
    for s in args.layer_insert:
        index, desc = s.split(":")
        index = int(index)
        if index > len(model.layers):
            raise Exception("Error: index %i too large (%i layers)"%(index, len(model.layers)))

        layers_before = list(model.layers[:index])
        layers_after = list(model.layers[index:])
        model.build_layer(desc, layers_before, args.activation, args.border_mode, args.weight_init)
        model.layers = layers_before + layers_after

    model_cnn.save_to_file(model, args.output)
    model_reload = True

if len(args.layer_append) > 0:
    if model_reload:
        model = model_cnn.load_from_file(args.output)

    print("Adding new layers: ", args.layer_append)
    for layer_desc in args.layer_append:
        model.build_layer(layer_desc, model.layers, args.activation, args.border_mode, args.weight_init)
    model_cnn.save_to_file(model, args.output)

print("--------FINAL MODEL---------")
model = model_cnn.load_from_file(args.output)
print("----------------------------")

for layer in model.layers:
    print(layer.type_name, " = ", sum([param.get_value(borrow=True, return_internal_type=True).size for param in layer.params()]))

print("Done")
