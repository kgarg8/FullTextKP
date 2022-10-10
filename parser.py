import argparse
from argparse import ArgumentParser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = ArgumentParser(description="Seq2Seq")
    parser.add_argument('--model', type=str, default='GRUSeq2Seq', choices=['GRUSeq2Seq', 'LEDSeq2Seq'])
    parser.add_argument('--model_type', type=str, default='Seq2Seq', choices=['Seq2Seq'])
    parser.add_argument('--dataset', type=str, default='ACM', choices=['ACM'])
    parser.add_argument('--test', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--initial_time', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--no_display', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--display_params', type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--example_display_step', type=int, default=500)
    parser.add_argument('--checkpoint', type=str2bool, help="checkpointing is done always whether you want it or not but this option can be used to load existing checkpoint", default=False)
    return parser