source configs_path.sh

python ../deep_pipe/experiment/build_experiment.py -cp $CONFIGS_PATH/train.config -ep ../exp/train
cd ../exp/train/experiment_0
snakemake
