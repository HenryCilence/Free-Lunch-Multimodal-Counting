from utils.cl_regression_trainer import RegTrainer
import argparse
import os
import torch

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data-dir', default=r"", help='training data directory')
    parser.add_argument('--save-dir', default='', help='directory to save models.')
    parser.add_argument('--lr', type=float, default=1e-5, help='the initial learning rate')
    parser.add_argument('--resume', default='', help='the path of resume training model')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--crop-size', type=int, default=224, help='default 224')
    parser.add_argument('--wd', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--max-model-num', type=int, default=1, help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1000, help='max training epoch')
    parser.add_argument('--warm-up-epoch', type=int, default=50, help='warm-up training epoch')
    parser.add_argument('--val-epoch', type=int, default=1, help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=1000, help='the epoch start to val')
    parser.add_argument('--save-all-best', type=bool, default=False, help='whether to load opt state')
    parser.add_argument('--batch-size', type=int, default=8, help='train batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='the num of training process')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
