#basic layer types
from denet.layer import IdentityLayer, InitialLayer
from denet.layer.convolution import ConvLayer
from denet.layer.deconvolution import DeconvLayer
from denet.layer.pool import PoolLayer
from denet.layer.pool_inv import PoolInvLayer
from denet.layer.batch_norm import BatchNormLayer
from denet.layer.batch_norm_relu import BatchNormReluLayer
from denet.layer.activation import ActivationLayer
from denet.layer.dropout import DropoutLayer
from denet.layer.resnet import ResnetLayer
from denet.layer.crop_mirror import CropMirrorLayer
from denet.layer.border import BorderLayer
from denet.layer.regression import RegressionLayer
from denet.layer.split import SplitLayer
from denet.layer.skip import SkipLayer, SkipSrcLayer
layer_types = [IdentityLayer, DropoutLayer, BorderLayer, ConvLayer, PoolLayer, PoolInvLayer,
               RegressionLayer, CropMirrorLayer, ActivationLayer, BatchNormLayer, BatchNormReluLayer, 
               ResnetLayer, DeconvLayer, SplitLayer, SkipLayer, SkipSrcLayer]

#DeNet detection layers
from denet.layer.denet_corner import DeNetCornerLayer
from denet.layer.denet_sparse import DeNetSparseLayer
from denet.layer.denet_detect import DeNetDetectLayer
layer_types += [DeNetCornerLayer, DeNetSparseLayer, DeNetDetectLayer]
