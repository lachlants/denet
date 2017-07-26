# DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling
By Lachlan Tychsen-Smith (research engineer), Lars Petersson (supervisor)
https://arxiv.org/abs/1703.10295

## Model Downloads
* DeNet-34 skip: [Google Drive](https://drive.google.com/uc?export=download&id=0B2Y3zi7OSEbrNzFKbXZDb2d1REE)
* DeNet-101 skip: [Google Drive](https://drive.google.com/uc?export=download&id=0B2Y3zi7OSEbrdE9MbERvUjNiV2s)

## Prediction
Use the following command to evaluation models on gpu0:

    GPU=0 model-predict --batch-size 8 --thread-num 4 --predict-mode detect,voc --input VOC_DIR --model MODEL_FNAME --extension voc,2007-test,images_per_subset=128,crop=512,scale_mode=large
    
where: 
* VOC_DIR is the path to the VOCdevkit directory
* MODEL_FNAME is the path to the downloaded model file (see above)  

To output images with annotated bounding boxes set "--predict-mode detect,voc,image"
