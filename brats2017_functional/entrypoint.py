from os.path import join
from functools import partial

import numpy as np
import torch.nn as nn
import nibabel as nib

from dpipe.medim import load_image
from dpipe.medim.bounding_box import extract, get_slice
from dpipe.torch.model import TorchFrozenModel
from dpipe.model_core.deepmedic_els import DeepMedicEls
from dpipe.batch_predict.patch_3d_fixed import Patch3DFixedPredictor

prefix = ''

data_path = prefix + '/input'
output_file = prefix + '/output/segm.nii.gz'

models_path = prefix + '/app/models'

input_files = [join(data_path, f) for f in ('t1.nii.gz', 't1c.nii.gz', 't2.nii.gz', 'fla.nii.gz')]


def load_input():
    return np.array([load_image(f) for f in input_files], dtype=np.float32)


def preprocess(x):
    x_bb, = extract([x], np.any(x > 0, axis=0))
    mean = x_bb.mean(axis=(1, 2, 3), keepdims=True)
    std = x_bb.std(axis=(1, 2, 3), keepdims=True)

    return np.array((x - mean) / std, dtype=np.float32)


def predict_inner(x, model_id):
    model = TorchFrozenModel(
        DeepMedicEls(4, 4, downsample='avg', upsample='neighbour', activation=nn.functional.relu), cuda=False,
        logits2pred=partial(nn.functional.softmax, dim=1), restore_model_path=join(models_path, f'model_{model_id}')
    )

    batch_predict = Patch3DFixedPredictor(x_patch_sizes=[[106, 106, 106], [138, 138, 138]], y_patch_size=[90, 90, 90])

    return batch_predict.predict(x, predict_fn=model.do_inf_step)


def predict(x, model_id):
    y_pred = np.zeros_like(x)
    y_pred[0] = 1
    inner_slice = [...] + get_slice(np.any(x > 0, axis=0))
    y_pred[inner_slice] = predict_inner(x[inner_slice], model_id)

    return y_pred


def save_result(y_proba):
    y_pred = np.argmax(y_proba, axis=0)
    # BraTS competition used this replacement
    # y_pred[y_pred == 3] = 4
    y_pred = y_pred.astype(np.uint8)
    img = nib.Nifti1Image(y_pred, np.eye(4))

    # Don't know if it's needed
    img.header.get_xyzt_units()

    nib.save(img, output_file)


if __name__ == '__main__':
    x = load_input()
    x = preprocess(x)

    y_proba = np.mean([predict(x, i) for i in (0, 6)], axis=0) 

    save_result(y_proba)
