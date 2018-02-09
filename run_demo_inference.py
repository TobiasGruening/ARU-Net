from __future__ import print_function, division

import click

from pix_lab.util.inference_pb import Inference_pb
from pix_lab.util.util import read_image_list

@click.command()
@click.option('--path_list_imgs', default="./demo_images/imgs.lst")
@click.option('--path_net_pb', default="./demo_nets/model100_ema.pb")
def run(path_list_imgs, path_net_pb):
    list_inf = read_image_list(path_list_imgs)
    inference = Inference_pb(path_net_pb, list_inf, mode='L')
    inference.inference()

if __name__ == '__main__':
    run()