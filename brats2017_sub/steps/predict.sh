source configs_path.sh
SCRIPTS_PATH="../../../deep_pipe/scripts"

cd ../exp/train/experiment_0/
python $SCRIPTS_PATH/predict.py -cp $CONFIGS_PATH/$TYPE.config --ids_path null --predictions_path $TYPE_pred_proba --restore_model_path model
python $SCRIPTS_PATH/transform.py -cp $CONFIGS_PATH/$TYPE.config --input_path $TYPE_pred_proba --output_path $TYPE_pred --transform pred2msegm
mkdir $TYPE_submission
python $SCRIPTS_PATH/make_brats_submit.py -pp $TYPE_pred -sp $TYPE_submission
mv $TYPE_submission ../../../
