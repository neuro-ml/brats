BRATS_PATH=$1
CONFIGS_PATH='configs'

cat <(echo -e 'data_path = "'$BRATS_PATH/train'"\n') $CONFIGS_PATH/generic.train.config $CONFIGS_PATH/generic.config > $CONFIGS_PATH/train.config
cat <(echo -e 'data_path = "'$BRATS_PATH/val'"\n') $CONFIGS_PATH/generic.val_test.config $CONFIGS_PATH/generic.config > $CONFIGS_PATH/val.config
cat <(echo -e 'data_path = "'$BRATS_PATH/test'"\n') $CONFIGS_PATH/generic.val_test.config $CONFIGS_PATH/generic.config > $CONFIGS_PATH/test.config
