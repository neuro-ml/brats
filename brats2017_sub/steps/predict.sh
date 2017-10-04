CONFIGS_PATH=../../../configs
SCRIPTS_PATH="../../../deep_pipe/scripts"

cd ../exp/train/experiment_0/
python $SCRIPTS_PATH/predict.py -cp "$CONFIGS_PATH/$TYPE.config" -ip null -pp $TYPE"_submission_pred_proba" --restore_model_path model
python $SCRIPTS_PATH/transform.py -cp "$CONFIGS_PATH/$TYPE.config" -ip $TYPE"_submission_pred_proba" -op $TYPE"_submission_pred" --transform pred2msegm
mkdir $TYPE"_submission"
python $SCRIPTS_PATH/make_brats_submit.py -pp $TYPE"_submission_pred" -sp $TYPE"_submission"
mv $TYPE"_submission" ../../../
