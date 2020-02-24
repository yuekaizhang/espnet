#!/usr/bin/env python3
from espnet2.tasks.pretrain import PretrainTask


def get_parser():
    parser = PretrainTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python pretrain.py asr --print_config --optim adadelta \
                > conf/pretrain.yaml
        % python pretrain.py --config conf/pretrain.yaml
    """
    PretrainTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
