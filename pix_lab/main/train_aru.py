from __future__ import print_function, division

import os
import click
from pix_lab.models.aru_net import ARUnet
from pix_lab.data_provider.data_provider_la import Data_provider_la
from pix_lab.training.trainer import Trainer

@click.command()
@click.option('--path_list_train', default="......./lists/train.lst")
@click.option('--path_list_val', default="......./lists/val.lst")
@click.option('--output_folder', default="......./models/")
@click.option('--restore_path', default=None)
def run(path_list_train, path_list_val, output_folder, restore_path):
    # Since the input images are of arbitrarily size, the autotune will significantly slow down training!
    # (it is calculated for each image)
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    # Images have to be gray scale images
    img_channels = 1
    # Number of output classes
    n_class = 3
    kwargs_dat = dict(batchsize_tr=1, scale_min=0.2, scale_max=0.5, scale_val=0.33, affine_tr=True,
                      one_hot_encoding=True)
    data_provider = Data_provider_la(path_list_train, path_list_val, n_class, kwargs_dat=kwargs_dat)

    # choose between 'u', 'ru', 'aru', 'laru'
    model_kwargs = dict(model="ru")
    model = ARUnet(img_channels, n_class, model_kwargs=model_kwargs)
    opt_kwargs = dict(optimizer="rmsprop", learning_rate=0.001)
    cost_kwargs = dict(cost_name="cross_entropy")
    trainer = Trainer(model,opt_kwargs=opt_kwargs, cost_kwargs=cost_kwargs)
    trainer.train(data_provider, output_folder, restore_path, batch_steps_per_epoch=256, epochs=100, gpu_device="0")


if __name__ == '__main__':
    run()