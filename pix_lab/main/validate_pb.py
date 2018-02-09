from __future__ import print_function, division

import click

from pix_lab.data_provider.data_provider_la import Data_provider_la
from pix_lab.util.validator_pb import Validator_pb


@click.command()
@click.option('--path_list_val', default="./demo_images/imgs.lst")
@click.option('--restore_pb_path', default="./demo_nets/model100_ema.pb")
def run(path_list_val, restore_pb_path):
    n_class = 3
    kwargs_dat=dict(scale_val=0.33, one_hot_encoding=True, shuffle=False)
    data_provider = Data_provider_la(None, path_list_val, n_class, kwargs_dat=kwargs_dat)
    cost_kwargs = dict(cost_name="cross_entropy")
    validator = Validator_pb(restore_pb_path, n_class=n_class, cost_kwargs=cost_kwargs)
    validator.validate(data_provider)


if __name__ == '__main__':
    run()