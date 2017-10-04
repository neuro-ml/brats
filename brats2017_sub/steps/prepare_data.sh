BRATS_PATH=$1

cd ../deep_pipe/dataset_preprocessing/brats2017
python make_metadata_train.py $BRATS_PATH/train
python make_metadata_val_test.py $BRATS_PATH/val
python make_metadata_val_test.py $BRATS_PATH/test
