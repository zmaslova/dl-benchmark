import argparse
import logging as log
import os
import sys
from pathlib import Path

from proc_utils import cmd_execution


def cli_argument_parser():
    parser = argparse.ArgumentParser(description='Inference launcher')
    parser.add_argument('-m', '--sync_mode',
                        type=str,
                        dest='sync_mode',
                        choices=['sync', 'async', 'tf'],
                        help='Inference mode to execute: sync, async or tensorflow (tf)')
    parser.add_argument('-ni', '--number_iter',
                        help='Number of inference iterations',
                        default=1,
                        type=int,
                        dest='number_iter')

    subparser = parser.add_subparsers(dest='inference')
    inference_args = subparser.add_parser('inference_args')

    inference_args.add_argument('inference_args',
                                nargs='+',
                                help='All necessary arguments for selected inference mode')

    args = parser.parse_args()

    return args


def local_start():
    log.basicConfig(
        format='[ %(levelname)s ] %(message)s',
        level=log.INFO,
        stream=sys.stdout,
    )
    args = cli_argument_parser()

    launchers = {'sync': os.path.join(Path(__file__).parent.parent, 'inference', 'inference_sync_mode.py'),
                 'async': os.path.join(Path(__file__).parent.parent, 'inference', 'inference_async_mode.py'),
                 'tf': os.path.join(Path(__file__).parent.parent, 'inference', 'inference_tensorflow.py')}
    launcher_path = launchers[args.sync_mode]

    log.info(f'Starting inference with {args.sync_mode} mode')

    exit_code, _ = cmd_execution(['python3', launcher_path] + args.inference_args, log=log)
    if exit_code != 0:
        log.error(f"Run of launcher with mode '{args.sync_mode}' failed with return code '{exit_code}'.")

    return exit_code


if __name__ == '__main__':
    sys.exit(local_start() or 0)
