import argparse
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from flowlane.utils.config import get_config
from flowlane.utils.py_config import Config
from flowlane.utils.setup import setup
from flowlane.engine.engine import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='FlowLane')
    parser.add_argument(
        '-c', '--config-file', metavar="FILE", help='config file path')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    # cuda setting
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    # checkpoint and log
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='put the path to resuming file if needed')
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='put the path to resuming file if needed')
    # for evaluation
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        default=False,
        help='skip validation during training')
    # for export
    parser.add_argument(
         '--export',
        type=str,
        default=None,
        help='put the path to resuming file if needed')
    parser.add_argument(
        '--export_repvgg',
        action='store_true',
        default=False,
        help='skip validation during training')
    # config options
    parser.add_argument(
        'opts',
        help='See config for all options',
        default=None,
        nargs=argparse.REMAINDER)

    #for inference
    parser.add_argument(
        "--source_path",
        default="",
        metavar="FILE",
        help="path to source image")
    parser.add_argument(
        "--reference_dir", default="", help="path to reference images")
    parser.add_argument("--model_path", default=None, help="model for loading")

    args = parser.parse_args()

    return args

def main(args, cfg):
    # init environment, include logger, dynamic graph, seed, device, train or test mode...
    setup(args, cfg)
    # build trainer
    trainer = Trainer(cfg)

    # continue train or evaluate, checkpoint need contain epoch and optimizer info
    if args.resume:
        trainer.resume(args.resume)
    # evaluate or finute, only load generator weights
    elif args.load:
        trainer.load(args.load)

    # export model to inference or pretrain weight form
    if args.export:
        trainer.export(args.export)
        return 

    if args.evaluate_only:
        trainer.val()
        return
        
    trainer.train()

if __name__ == '__main__':
    args = parse_args()
    if args.config_file.endswith('yaml') or args.config_file.endswith('yml'):
        cfg = get_config(
            args.config_file, overrides=args.override)
    elif args.config_file.endswith('py'):
        cfg = Config.fromfile(args.config_file)
    main(args, cfg)