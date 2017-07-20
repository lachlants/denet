Paper Details:
* DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling
* Lachlan Tychsen-Smith (research engineer), Lars Petersson (supervisor)
* https://arxiv.org/abs/1703.10295

These bash scripts can be used to recreate the mscoco and pascal voc experiments in the paper:
* denet34.sh: Trains and predicts a DeNet34 standard/skip/wide detector on Pascal VOC or MSCOCO using 2-4x GPUs 
* denet101.sh: Trains and predicts a DeNet101 standard/skip/wide detector on Pascal VOC or MSCOCO using 4x GPUs 

Both script arguments take the form: script.sh MODEL_VARIANT DATA_TYPE DATA_DIR
where
* MODEL_VARIANT is either "std", "skip" or "wide"
* DATA_TYPE is either "mscoco", "voc2007" or "voc2012"
* DATA_DIR is the directory containing the relevant dataset e.g. for voc2007/voc2012 we require the 'VOCdevkit' directory containing both extracted datasets, for mscoco we need the directory containing the extracted "annotations", "test2015", "train2014" and "val2014" directories. 

Requires the resnet34.mdl.gz and resnet101.mdl.gz models are present in the models/ directory. 
DeNet101 models require GPUs with at least 12GB of RAM.