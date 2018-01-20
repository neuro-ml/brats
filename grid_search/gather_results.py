import os
import argparse
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm

from dpipe.io import load_json

valuable_params = ['batch_size', 'n_iters_per_epoch', 'n_epochs', 'downsample', 'nonzero_fraction']

valuable_metrics = ['mean', 'median', 'std']


def load_result(experiment_path):
    result = {}
    for fold_name in os.listdir(experiment_path):
        if fold_name.startswith('experiment'):
            result.update(load_json(join(experiment_path, fold_name, 'test_dices.json')))

    return np.array(list(result.values()))


def gather_results(experiments_path):
    records = {}

    for experiment_name in tqdm(os.listdir(experiments_path)):
        experiment_path = join(experiments_path, experiment_name)
        if not os.path.isdir(experiment_path) or experiment_name.startswith('.'):
            continue

        result = load_result(experiment_path)

        # rm = get_resource_manager(join(experiment_path, 'config'))
        # record = {param: getattr(rm, param) for param in valuable_params}

        record = {}
        for metric_name in valuable_metrics:
            metric_value = getattr(np, metric_name)(result, axis=0)
            for i, value in enumerate(metric_value):
                record[f'dice{i}_{metric_name}'] = value

        records[experiment_name] = record

    df = pd.DataFrame.from_dict(records, orient='index')
    return df[sorted(df.columns.tolist())]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments_path')
    args = parser.parse_known_args()[0]

    df = gather_results(args.experiments_path)
    df.to_csv('results.csv', index_label='experiment_name')
