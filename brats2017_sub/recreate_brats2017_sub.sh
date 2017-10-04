$BRATS_PATH = $1

cd steps
./install_deep_pipe.sh
./prepare_data.sh $BRATS_PATH
./build_configs.sh
./prepare_brats2017.sh $BRATS_PATH
./trains.sh