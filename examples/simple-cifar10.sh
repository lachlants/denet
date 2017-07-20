#Example: Trains a simple network with dropout and batch normalization on CIFAR10 dataset using single GPU

TRAIN_DIR=$1
VAL_DIR=$2
OUTPUT_DIR=./simple-cifar10-model
MODEL_DESC="B[3] C[128,3] BN A C[96,2] BN A C[64,1] BN A P.A[2] D[0.2] C[256,3] BN A C[192,2] BN A C[128,1] BN A P.A[2] D[0.2] C[512,3] BN A C[384,2] BN A C[256,1] BN A D[0.2] R.C[6]"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIN=$SCRIPT_DIR"/../bin/model-train"

#check parameters
if [ -z "$GPU" ]; then
    echo "Environment variable GPU not specified assuming GPU=0"
    export GPU=0
fi

if [ ! -d "$TRAIN_DIR" ] || [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: Please provide CIFAR10 training and validation directories on cmdline"
    exit 1
fi

#make output directories
mkdir -p $OUTPUT_DIR

#print some stuff
echo "Training simple model on cifar10..."
echo "- Train Dir: $TRAIN_DIR"
echo "- Test Dir: $VAL_DIR"
echo "- Output Dir: $OUTPUT_DIR"
echo "- Logs: $OUTPUT_DIR/train.[out|err]"
echo "- Binary: $BIN"

cd $OUTPUT_DIR

echo "Running training..."
$BIN --seed 0 --distort-mode o4 --solver sgd --border-mode same --augment-mirror --activation relu --epochs 90 --batch-size 32 --train "$TRAIN_DIR" --test "$VAL_DIR" --extension png --learn-rate 0.1 --learn-momentum 0.9 --learn-anneal 0.5 --learn-anneal-epochs 15 30 45 60 75 --learn-decay 0.0005 --model-desc $MODEL_DESC > train.out 2> train.err

echo "Done"
