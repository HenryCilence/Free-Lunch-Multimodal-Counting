import torch
from torch.optim.lr_scheduler import LambdaLR
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import time
from torch import optim
from torch.utils.data import DataLoader
import logging
import numpy as np
from tqdm import tqdm
from datasets.crowd import Crowd
from models.moco import PPCA_RGBT as Train_Model


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    if type(transposed_batch[0][0]) == list:
        rgb_list = [item[0] for item in transposed_batch[0]]
        t_list = [item[1] for item in transposed_batch[0]]
        rgbt_list = [item[2] for item in transposed_batch[0]]
        rgb = torch.stack(rgb_list, 0)
        t = torch.stack(t_list, 0)
        rgbt = torch.stack(rgbt_list, 0)
        images = [rgb, t, rgbt]
    else:
        images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    st_sizes = torch.FloatTensor(transposed_batch[2])
    return images, points, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.datasets = Crowd(os.path.join(args.data_dir, 'train'), args.crop_size, 8, 'train')
        self.dataloaders = DataLoader(self.datasets,
                                      collate_fn=train_collate,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers * self.device_count,
                                      pin_memory=True,
                                      drop_last=True)

        self.model = Train_Model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

        # Learning rate scheduler: warm-up and cosine annealing
        def lr_lambda(epoch):
            if epoch < args.warm_up_epoch:
                return float(epoch + 1) / float(max(1, args.warm_up_epoch))
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warm_up_epoch) / (args.max_epoch - args.warm_up_epoch)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.scheduler.last_epoch = self.start_epoch - 1  # Ensure scheduler is in sync with the optimizer
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_loss = np.inf
        self.best_loss_epoch = -1

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            self.train_epoch()
            self.scheduler.step()  # Update the learning rate

    def train_epoch(self):
        epoch_start = time.time()
        epoch_loss = AverageMeter()
        self.model.train()  # Set model to training mode

        dataloader = tqdm(self.dataloaders, desc="Training", leave=False, dynamic_ncols=True)
        # Iterate over data.
        for step, (inputs, _, _) in enumerate(dataloader):

            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
                inputs[2] = inputs[2].to(self.device)
            else:
                inputs = inputs.to(self.device)

            with torch.set_grad_enabled(True):
                loss = self.model(inputs[0], inputs[1], inputs[2])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if type(inputs) == list:
                    N = inputs[0].size(0)
                else:
                    N = inputs.size(0)

                epoch_loss.update(loss.item(), N)

        dataloader.close()

        loss_item = epoch_loss.get_avg()
        if loss_item < self.best_loss:
            self.best_loss = loss_item
            self.best_loss_epoch = self.epoch
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models

        logging.info('Epoch {} Train, Loss: {:.4f}, Best Loss: {:.4f} on epoch {}, lr: {:.4f}x, Cost {:.1f} sec'
                     .format(self.epoch, loss_item, self.best_loss, self.best_loss_epoch,
                             self.optimizer.param_groups[0]['lr'] / self.args.lr, time.time() - epoch_start))
