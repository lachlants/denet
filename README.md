#DeNet
-----------
A simple extendable library for training and evaluating Deep Convolutional Neural Networks focussing on image classification and detection. 

Requirements
-----------
* python3
* numpy
* pillow
* scipy
* theano REQUIRES v8.1 (doesn't work with new gpuarray interface in v9.0)
* cudnn 

Features
-----------
* Reference "DeNet: Scalable real-time object detection with directed sparse sampling" implementation (see /papers/dss/)
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

Contact
-----------
* Lachlan Tychsen-Smith (Research Engineer @ Data61, CSIRO, Australia) - lachlan.ts@gmail.com or Lachlan.Tychsen-Smith@data61.csiro.au


