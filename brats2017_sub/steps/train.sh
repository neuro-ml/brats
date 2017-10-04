python ../deep_pipe/experiment/build_experiment.py -cp configs/train.config -ep ../exp/train
cd ../exp/train/experiment_0
snakemake
