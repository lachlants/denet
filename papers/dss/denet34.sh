#!/bin/bash
# Paper Details:
# - DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling
# - Lachlan Tychsen-Smith (research engineer), Lars Petersson (supervisor)
# - https://arxiv.org/abs/1703.10295

# Trains and predicts a DeNet34 standard/skip/wide detector on Pascal VOC or MSCOCO using 2-4x GPUs 

MODEL_VAR=$1
if [[ $MODEL_VAR == "std" ]]; then
    MODEL_DESC="PI[2] C.B[256,3] BNA PI[2] C.B[128,3] BNA DNC[96,100] DNS[7,24,0.01,0.1] C.B[1536,1] BNA C.B[1024,1] BNA C.B[768,1] BNA C.B[512,1] BNA DND[0.5,1,1]"
    BSF=2
elif [[ $MODEL_VAR == "skip" ]]; then
    MODEL_DESC="PI[2] C[256,3] SKIP[1] BNA PI[2] C[128,3] SKIP[0] BNA DNC[96,100] DNS[7,24,0.01,0.1] C[1536,1] BNA C.B[1024,1] BNA C.B[768,1] BNA C.B[512,1] BNA DND[0.5,1,1]"
    BSF=2
elif [[ $MODEL_VAR == "wide" ]]; then
    MODEL_DESC="PI[2] C[256,3] SKIP[2] BNA PI[2] C[128,3] SKIP[1] BNA PI[2] C[64,3] SKIP[0] BNA SPLIT DNC[48,400] DNS[10,48,0.01,0.1] C.B[1536,1] BNA C.B[1024,1] BNA C.B[768,1] BNA C.B[512,1] BNA DND[0.5,1,1]"
    BSF=1
else
    echo "ERROR: Please provide model variant (std,skip,wide) on cmdline (arg 1)"
    exit 1
fi

DATASET=$2
if [[ $DATASET != "voc2007" &&  $DATASET != "voc2012" && $DATASET != "mscoco" ]]; then
    echo "ERROR: Please provide dataset type=(voc2007,voc2012,mscoco) on cmdline (arg 2)"
    exit 1
fi

INPUT_DIR=$3
if [ ! -d "$INPUT_DIR" ]; then
    if [[ $DATASET == "voc2007" || $DATASET == "voc2012" ]]; then 
	echo "ERROR: Please provide Pascal 'VOCdevkit' directory with 2007 / 2012 datasets extracted on cmdline (arg 3)"
    else
	echo "ERROR: Please provide MSCOCO directory containing annotations, train2014, etc with 2014 / 2015 datasets extracted on cmdline (arg 3)"
    fi
    exit 1
fi

DENET_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../../"
OUTPUT_DIR=./denet34-$DATASET-$MODEL_VAR
IMAGE_LOADER="images_per_subset=1280,scale=512,crop=512,augment_photo,crop_mode=denet,scale_mode=large"
TRAIN_PARAM="--solver nesterov --epochs 90 --batch-size 32 --batch-size-factor $BSF --learn-rate 0.1 --learn-momentum 0.9 --learn-anneal 0.1 --learn-anneal-epochs 30 60 --learn-decay 0.0001"

if [[ $DATASET == "voc2007" ]]; then
    DATA_TYPE=voc
    TRAIN_DATA=2007-trainval,2012-trainval
    TEST_DATA=2007-test
    CLASS_NUM=20
elif [[ $DATASET == "voc2012" ]]; then
    DATA_TYPE=voc
    TRAIN_DATA=2007-trainvaltest,2012-trainval
    TEST_DATA=2012-test
    CLASS_NUM=20
else 
    DATA_TYPE=mscoco
    TRAIN_DATA=2014-train,2014-val
    TEST_DATA=2015-test
    CLASS_NUM=80
fi

#set to valid gpus (caution! changing number of GPUS changes batch size)
if [[ $MODEL_VAR == "wide" ]]; then
    GPUS="gpu0 gpu1 gpu2 gpu3"
else
    GPUS="gpu0 gpu1"
fi

#make output directories
mkdir -p $OUTPUT_DIR

#print some stuff
echo "Training DeNet-34 ($MODEL_VAR) model on $DATASET with $GPUS"
echo "- Input Dir: $INPUT_DIR"
echo "- Output Dir: $OUTPUT_DIR"
echo "- Class Num: $CLASS_NUM"
echo "- Train Data: $TRAIN_DATA"
echo "- Test Data: $TEST_DATA"
echo ""

cd $OUTPUT_DIR

#modify base model
if [ ! -f "./initial.mdl.gz" ]; then
    echo "Modifying base model..."
    if [[ $MODEL_VAR == "skip" ]]; then
	$DENET_DIR/bin/model-modify --input $DENET_DIR/models/imagenet/resnet34.mdl.gz --output initial_skipsrc.mdl.gz --modify-bn 1 0.9 1e-5 --convert-bn-relu --use-cudnn-pool  --class-num $CLASS_NUM --image-size 512 512 --layer-remove 3 --layer-insert 11:SKIPSRC.X[0] 18:SKIPSRC.X[1] > ./modify_skipsrc.out 2> modify_skipsrc.err
	$DENET_DIR/bin/model-modify --input initial_skipsrc.mdl.gz --output initial.mdl.gz --layer-append $MODEL_DESC > ./modify.out 2> modify.err

    elif [[ $MODEL_VAR == "wide" ]]; then
	$DENET_DIR/bin/model-modify --input $DENET_DIR/models/imagenet/resnet34.mdl.gz --output initial_skipsrc.mdl.gz --modify-bn 1 0.9 1e-5 --convert-bn-relu --use-cudnn-pool  --class-num $CLASS_NUM --image-size 512 512 --layer-remove 3 --layer-insert 7:SKIPSRC[0] 12:SKIPSRC.X[1] 19:SKIPSRC.X[2] > ./modify_skipsrc.out 2> modify_skipsrc.err
	$DENET_DIR/bin/model-modify --input initial_skipsrc.mdl.gz --output initial.mdl.gz --layer-append $MODEL_DESC > ./modify.out 2> modify.err

    else
	$DENET_DIR/bin/model-modify --input $DENET_DIR/models/imagenet/resnet34.mdl.gz --output initial.mdl.gz --modify-bn 1 0.9 1e-5 --convert-bn-relu --use-cudnn-pool --class-num $CLASS_NUM --image-size 512 512 --layer-remove 3 --layer-append $MODEL_DESC > ./modify.out 2> modify.err
    fi
fi

#first execution just generates model-dims.json!
if [ ! -f "./model-dims.json" ]; then
    echo "Constructing model-dims.json..."
    $DENET_DIR/bin/model-train-multi $TRAIN_PARAM --gpus $GPUS --thread-num 4 --seed 1 --train "$INPUT_DIR" --extension $DATA_TYPE,$TRAIN_DATA,$IMAGE_LOADER --model ./initial.mdl.gz > model-dims.out 2> model-dims.err
fi

echo "Running training..."
$DENET_DIR/bin/model-train-multi $TRAIN_PARAM --gpus $GPUS --thread-num 4 --seed 1 --train "$INPUT_DIR" --extension $DATA_TYPE,$TRAIN_DATA,$IMAGE_LOADER --model ./initial.mdl.gz > train.out 2> train.err

#
mkdir predict
cd predict

echo "Merging final Model..."
$DENET_DIR/bin/model-modify --input ../model_epoch089_final.mdl.gz --output merged.mdl.gz --merge > ./merge.out 2> merge.err

echo "Running prediction on gpu0..."
GPU=0 $DENET_DIR/bin/model-predict --predict-mode detect,$DATA_TYPE --batch-size 8 --thread-num 4 --model ./merged.mdl.gz --input "$INPUT_DIR" --extension $DATA_TYPE,$TEST_DATA,$IMAGE_LOADER > predict.out 2> predict.err

echo "Done"
