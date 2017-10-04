BRATS_PATH=$1

cd steps
./install_deep_pipe.sh
./prepare_data.sh $BRATS_PATH
./build_configs.sh $BRATS_PATH
./train.sh
./predict_val.sh
./predict_test.sh
