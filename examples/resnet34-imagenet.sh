#Example: Trains a ResNet-34 (B) on ImageNet dataset using 2x GPUs 
#See "Deep Residual Learning for Image Recognition" paper 

TRAIN_DIR=$1
VAL_DIR=$2
OUTPUT_DIR=./resnet34-imagenet-model
MODEL_DESC="C.B[64,7,2] BN A P[3,2,1] nRSN.O[3,64,3] nRSN.O[4,128,3,2] nRSN.O[6,256,3,2] nRSN.O[3,512,3,2] P.A[7] R.TB"
IMAGE_LOADER="images_per_subset=12800,scale=256,crop=224,crop_mode=lenet,scale_mode=small,augment_color,augment_photo"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIN=$SCRIPT_DIR"/../bin/model-train-multi"

#set to valid gpus (caution! changing number of GPUS changes batch size)
GPUS="gpu0 gpu1"

#check parameters
if [ ! -d "$TRAIN_DIR" ] || [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: Please provide ImageNet training and validation directories on cmdline"
    exit 1
fi

#make output directories
mkdir -p $OUTPUT_DIR

#print some stuff
echo "Training RESNET-34 (B) model on ImageNet..."
echo "- Train Dir: $TRAIN_DIR"
echo "- Test Dir: $VAL_DIR"
echo "- Output Dir: $OUTPUT_DIR"
echo "- Logs: $OUTPUT_DIR/train.[out|err]"
echo "- Binary: $BIN"

cd $OUTPUT_DIR

#first execution just generates model-dims.json!
if [ ! -f "$OUTPUT_DIR/model-dims.json" ]; then
    echo "Constructing model-dims.json..."
    $BIN --solver torch --gpus $GPUS --thread-num 4 --seed 1 --epochs 90 --batch-size 64 --batch-size-factor 2 --train "$TRAIN_DIR" --test "$VAL_DIR" --extension imagenet,$IMAGE_LOADER --learn-rate 0.1 --learn-momentum 0.9 --learn-anneal 0.1 --learn-anneal-epochs 30 60 --learn-decay 0.0001 --model-desc $MODEL_DESC > model-dims.out 2> model-dims.err
fi

echo "Running training..."
$BIN --solver torch --gpus $GPUS --thread-num 4 --seed 1 --epochs 90 --batch-size 64 --batch-size-factor 2 --train "$TRAIN_DIR" --test "$VAL_DIR" --extension imagenet,$IMAGE_LOADER --learn-rate 0.1 --learn-momentum 0.9 --learn-anneal 0.1 --learn-anneal-epochs 30 60 --learn-decay 0.0001 --model-desc $MODEL_DESC > train.out 2> train.err

echo "Done"
