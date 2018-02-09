from __future__ import print_function, division

import os

import click

from pix_lab.data_provider.data_provider_la import Data_provider_la
from pix_lab.models.aru_net import ARUnet
from pix_lab.util.validator_ckpt import Validator_ckpt


@click.command()
@click.option('--path_list_val', default="./demo_images/imgs.lst")
@click.option('--restore_ckt_path', default="......./models/model100")
@click.option('--restore_ema', default=True)
def run(path_list_val, restore_ckt_path, restore_ema):
    # Since the input images are of arbitrarily size, the autotune will significantly slow down the training!
    # (it is calculated for each image)
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    img_channels = 1
    n_class = 3
    kwargs_dat=dict(scale_val=0.33, one_hot_encoding=True, shuffle=False)
    data_provider = Data_provider_la(None, path_list_val, n_class, kwargs_dat=kwargs_dat)
    model_kwargs = dict(model="ru")
    model = ARUnet(img_channels, n_class, model_kwargs=model_kwargs)
    cost_kwargs = dict(cost_name="cross_entropy")
    validator = Validator_ckpt(model, cost_kwargs=cost_kwargs)
    validator.validate(data_provider, restore_ckt_path, restore_ema)


if __name__ == '__main__':
    run()