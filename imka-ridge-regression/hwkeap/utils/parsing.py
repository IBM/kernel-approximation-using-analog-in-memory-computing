import argparse

from easydict import EasyDict as edict


def parse_args_simulations():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",help="Config file",required=False,default="experiments/simulations/config/config_attn.yml",type=str,)
    parser.add_argument("-m","--metric",help="Evaluation metric",required=False,default="app_err",type=str,)
    return edict(vars(parser.parse_args()))


def parse_args_hardware():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",help="Config file",required=False,default="experiments/hardware/config/config_cod-rna.yml",type=str,)
    return edict(vars(parser.parse_args()))
