import os.path as osp
import argparse
from yacs.config import CfgNode as CN

cfg = CN()

cfg.common = CN()
cfg.common.device = 'cuda'
cfg.common.model_path = None
cfg.common.model_name = None
cfg.common.ckpt_path = None
cfg.common.input_shape = None

cfg.model = CN()


def get_cfg_defaults():
    return cfg.clone()

def update_cfg(cfg_file):
    assert osp.exists(cfg_file), "Config file does not exist."
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='path to config')

    args = parser.parse_args()

    cfg_file = args.cfg

    if args.cfg is not None:
        cfg = update_cfg(cfg_file)
    else:
        cfg = get_cfg_defaults()

    return cfg


def load_args(cfg_file):
    cfg = update_cfg(cfg_file)
    return cfg