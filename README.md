DeNet
-----------
A simple extendable library for training and evaluating Deep Convolutional Neural Networks focussing on image classification and detection. 

Requirements
-----------
* python3
* numpy
* pillow
* scipy
* theano REQUIRES development version (see below)
* cudnn 

Installation
-----------

DeNet currently requires the development version of Theano:
1. Download Theano git e.g.

    git clone https://github.com/Theano/Theano.git

2. Revert to revision used in the papers (might not be neccessary):

    git checkout fadc8be492628dcef4760ab4cfcd147824a8226f

3. Setup PYTHONPATH environment variable:

    export PYTHONPATH=theano_git_dir:$PYTHONPATH

Probably want to add this command to your .bashrc so it doesn't have to be run repetitively.

Features
-----------
* Reference "DeNet: Scalable real-time object detection with directed sparse sampling" implementation (see /papers/dss/)
* Reference "Improving Object Localization with Fitness NMS and Bounded IoU Loss" implementation
* GPU optimized sparse sampling op and pool inv op
* Ability to split model into multiple executions for training (useful for very large models)
* Training on Multiple GPU's

Training
-----------
Training neural networks is performed via either "bin/model-train" or "bin/model-train-multi" for 
single / multi GPU implementation. Arguments are mostly the same for both implementations. The GPU to 
use for "model-train" is specified by exporting the "GPU=N" global variable (see examples), 
for "model-train-multi" GPUs are specified with the "--gpus" argument.

Note: the initial run of "model-train-multi" simply produces a "model-dims.json" file... you must
rerun the command to begin training!

Example - Training simple 3 layer CNN on CIFAR10 / CIFAR100: 

	GPU=0 bin/model-train --border-mode half --activation relu  --model-desc C[128,3] BN A P[2] C[256,3] BN A P[2] C[512,3] BN A P.A R  --learn-rate 0.1 --solver sgd --epochs 30 --train TRAIN --test TEST --extension png 

where TRAIN and TEST are the train and test folders for CIFAR10 / CIFAR100

NOTE: see examples/ for a more comprehensive list of examples

Datasets:
-----------
Datasets are specified using the "--train FNAME" and "--test FNAME" arguments. 

A number of different formats are supported including:
* Basic Dir: --ext=image type
  assumes directory is in format FNAME/class name/image.ext e.g. CIFAR10/CIFAR100
* Basic Array: --ext=npy
  uses FNAME_data.npy and FNAME_labels.npy saved numpy arrays. The data is of dim
  (B,C,H,W) and dtype=float32 and the labels is of dimension (B,) and dtype=int32 where B is the number of 
  samples in the dataset, C is the number of channels and WxH are the spatial dimensions. 
* ImageNet: --ext=imagenet,<image_loader arguments> 
* MSCOCO: --ext=mscoco,<image_loader arguments>
* Pascal VOC: --ext=voc,<image_loader arguments>

The ImageNet, MSCOCO and Pascal VOC share the image_loader.py interface which enables background loading of 
the dataset and additional augmentation schemes. For large datasets I recommend a similar implementation.

Augmentation is either specified via the image_loader interface (occuring as the image is loaded
from disk) or with the "--distort-mode" argument (operates on preloaded images). The basic distort-mode 
arguments are: 
* random offsets - oX where X is the number of pixels (X and Y) to randomly offset sample
* random scale - sX where X is percent to vary the scale by
* random rotation - rX where X is the degrees to rotation left/right
* random mirroring - mX where X is the percentage probabilty of applying mirroring

Models Desc:
-----------
Model layers are specified in a feedforward manner via the "--model-desc" argument and take the 
format "TYPE.TAGS[ARGS]" where TYPE defines the layer type and TAGS are optional single character 
switchs (specific to each type) and ARGS defines the arguments to supply (if any). In most cases 
not all ARGS must be specified! 

Common Layer Types:
* Convolution: C[filters, size, stride] or C.B to disable bias (useful for batch normalization)
  - filters = number of output filters, number of input filters is automatically determined
  - size = spatial dimensions of the convolution (default=1)
  - stride = the number of pixels between each evaluation (default=1) 
  - e.g. C.B[256,3,2] = 256x3x3 convolution without bias with a stride of 2x2
* Pooling: P[size, stride, pad] for "Max Pooling" or P.A for "Mean Pooling"
  - size = spatial dimensions of the pooling operation (default=input spatial size) 
  - stride = the number of pixels between each evaluation (default=size) 
  - pad = input padding (default=0) 
* Batch Normalization: BN[momentum, eps]
  - momentum = running mean momentum (default=0.9)
  - eps = epsilon to fix divide by zero errors (default=1e-5)
* Dropout: D[rate]
  - rate = probability of dropping an activation (default=0.5)
* Activation: A (applies activation of type specified in "--activation" argument)
* Regression: R (performs negative log-likelihood softmax regression to target classes)
* Residual Network Block: RSN[filters, size, stride]
* Multi Residual Network Blocks: nRSN[num, filters, size, stride]

Please refer to parse_desc() in the layer .py files for all arguments e.g. see /src/layer/batch_norm.py

Pretrained Models
-----------

See [MSCOCO](/models/mscoco/README.md) and [VOC2007](/models/voc2007/README.md) to download pretrained DeNet models.

### MSCOCO Results:

| Model                    | Eval. Rate | MAP@IoU=[0.5:0.95] | MAP@IoU=0.5 | MAP@IoU=0.75 |
| ------------------------ |:----------:|:------------------:|:-----------:|:------------:|
| DeNet34 skip (v1)        |    82 Hz   |        29.5%       |    47.7%    |     31.1%    |
| DeNet34 wide (v1)        |    44 Hz   |        30.0%       |    48.9%    |     31.8%    |
| DeNet101 skip (v1)       |    33 Hz   |        32.3%       |    51.4%    |     34.6%    |
| DeNet101 wide (v1)       |    17 Hz   |        33.8%       |    53.4%    |     36.1%    |
| DeNet34 wide (v2)        |    80 Hz   |        33.4%       |    49.8%    |     35.8%    |
| DeNet101 wide (v2)       |    21 Hz   |        36.8%       |    53.9%    |     39.3%    |
| DeNet101 wide x768 (v2)  |    11 Hz   |        39.5%       |    58.0%    |     42.6%    |

v1 models are from the original paper

v2 models are from the 2nd paper and include Joint fitness NMS, bounded IoU loss and corner clustering

The DeNet101wide x768 (v2) model is the DeNet101 wide (v2) model evaluated with 768x768 pixel input images and
1296 sample RoIs

Evaluation rate is for Titan X (Maxwell) GPU, Cuda v8.0 and CuDNN v5110. Note that variations in these versions
can cause the MAP to fluctuate a bit e.g. +- 0.2%

Modifying models
-----------
Use the model-modify application to modify parameters for existing models and layers. 
e.g. to run prediction with 768x768 input images and 1296 RoIs use:

    model-modify --input MODEL --output MODEL_X768 --image-size 768 768 --modify-layer denet-sparse sample_num=36
    GPU=0 model-predict --batch-size 8 --thread-num 4 --predict-mode detect,mscoco --input MSCOCO_DIR --model MODEL_X768 --extension mscoco,2015-test-dev,images_per_subset=128,crop=768,scale_mode=large

Note the change to the "crop" parameter in the model-predict extension argument. 

Contact
-----------
* Lachlan Tychsen-Smith (Research Engineer @ Data61, CSIRO, Australia) - lachlan.ts@gmail.com or Lachlan.Tychsen-Smith@data61.csiro.au


