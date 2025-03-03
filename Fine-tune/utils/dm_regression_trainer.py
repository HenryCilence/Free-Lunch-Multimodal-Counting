from utils.evaluation import eval_game, eval_relative
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
from models.counting.swin_unet import Swin_BM_RGBT, count_parameters

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.dm_crowd import Crowd
from datasets.crowd import Crowd as Test_Crowd
from losses.ot_loss import OT_Loss
from losses.LRD import CL1


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    if type(transposed_batch[0][0]) == list:
        rgb_list = [item[0] for item in transposed_batch[0]]
        t_list = [item[1] for item in transposed_batch[0]]
        rgb = torch.stack(rgb_list, 0)
        t = torch.stack(t_list, 0)
        images = [rgb, t]
    else:
        images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    gt_discretes = torch.stack(transposed_batch[2], 0)
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, gt_discretes, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            logging.info(f"Current torch seed: {torch.initial_seed()}, Current torch.cuda seed: {torch.cuda.initial_seed()}")
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = Crowd(os.path.join(args.data_dir, 'train'), args.crop_size, args.downsample_ratio, 'train')
        self.dataloader = DataLoader(self.datasets, collate_fn=train_collate, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers*self.device_count, pin_memory=True)

        self.val_dataset = Test_Crowd(os.path.join(args.data_dir, 'val'), method='val')
        self.val_dataloader = DataLoader(self.val_dataset, 1, shuffle=False, num_workers=8, pin_memory=True)

        self.test_dataset = Test_Crowd(os.path.join(args.data_dir, 'test'), method='test')
        self.test_dataloader = DataLoader(self.test_dataset, 1, shuffle=False, num_workers=8, pin_memory=True)

        self.model = Swin_BM_RGBT()
        self.model.to(self.device)

        logging.info('Total Trainable Params: {} M'.format(count_parameters(self.model)/1e6))
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.ot_loss = OT_Loss(args.crop_size, self.downsample_ratio, args.norm_cood, self.device,
                               args.num_of_iter_in_ot,
                               args.reg)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.count_loss = nn.L1Loss(reduction='sum').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.rd_loss = CL1()

        self.val_best_mae = np.inf
        self.val_best_mse = np.inf
        self.best_game = [np.inf, np.inf, np.inf, np.inf]
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                mae_is_best, mse_is_best = self.val_epoch()
            if epoch >= args.val_start and (mse_is_best or mae_is_best):  # or (epoch > 200 and epoch % 5 == 0)):
                self.test_epoch()

    def train_eopch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_game = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_rd_loss = AverageMeter()
        epoch_start = time.time()
        self.model.train()

        # Iterate over data.
        dataloader = tqdm(self.dataloader, desc="Training", leave=False, dynamic_ncols=True)
        for step, (inputs, points, gt_discrete, st_sizes) in enumerate(dataloader):

            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            if type(inputs) == list:
                N = inputs[0].size(0)
            else:
                N = inputs.size(0)

            with torch.set_grad_enabled(True):
                rgb, t = inputs
                outputs, features = self.model(rgb, t)
                outputs_sum = outputs.view([outputs.size(0), -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                outputs_normed = outputs / (outputs_sum + 1e-6)

                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points)
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)

                # Compute counting loss.
                count_loss = self.mae(outputs.sum(1).sum(1).sum(1),
                                      torch.from_numpy(gd_count).float().to(self.device))
                epoch_count_loss.update(count_loss.item(), N)

                # Compute TV loss.
                gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(
                    2).unsqueeze(3)
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(
                    1) * torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.wtv
                epoch_tv_loss.update(tv_loss.item(), N)

                loss = ot_loss + count_loss + tv_loss
                
                # Compute RD loss.
                rd_loss = self.rd_loss(features, points)
                
                total_loss = loss + rd_loss * self.args.wrd

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_rd_loss.update(rd_loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_game.update(np.mean(abs(pred_err)), N)
        dataloader.close()

        logging.info('Epoch {} Train, Counting Loss: {:.2f}, RD Loss: {:.4f}, GAME0: {:.2f} MSE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), epoch_rd_loss.get_avg(), epoch_game.get_avg(), np.sqrt(epoch_mse.get_avg()),
                             time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        self.model.eval()
        epoch_start = time.time()
        total_relative_error = 0
        epoch_res = []
        dataloader = tqdm(self.val_dataloader, desc="Validating", leave=False, dynamic_ncols=True)
        for inputs, target, name in dataloader:
            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)
    
            if len(inputs[0].shape) == 5:
                inputs[0] = inputs[0].squeeze(0)
                inputs[1] = inputs[1].squeeze(0)
            if len(inputs[0].shape) == 3:
                inputs[0] = inputs[0].unsqueeze(0)
                inputs[1] = inputs[1].unsqueeze(0)
    
            with torch.set_grad_enabled(False):
                rgb, t = inputs
                outputs, _ = self.model(rgb, t)
                res = torch.sum(target).item() - torch.sum(outputs).item()
                epoch_res.append(res)
    
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error
        dataloader.close()
    
        N = len(self.val_dataloader)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        mae_is_best = mae < self.val_best_mae
        mse_is_best = mse < self.val_best_mse
        total_relative_error = total_relative_error / N
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Re: {:.4f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, total_relative_error, time.time() - epoch_start))
    
        if mae_is_best or mse_is_best:
            self.val_best_mse = mse
            self.val_best_mae = mae
            logging.info("*** Best mse {:.2f} mae {:.2f} model epoch {}".format(self.val_best_mse,
                                                                                 self.val_best_mae,
                                                                                 self.epoch))
    
        return mae_is_best, mse_is_best

    def test_epoch(self):
        epoch_start = time.time()
        args = self.args
        self.model.eval()
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        dataloader = tqdm(self.test_dataloader, desc="Testing", leave=False, dynamic_ncols=True)
        i = 0
        for inputs, target, name in dataloader:
            i += 1
            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)

            if len(inputs[0].shape) == 5:
                inputs[0] = inputs[0].squeeze(0)
                inputs[1] = inputs[1].squeeze(0)
            if len(inputs[0].shape) == 3:
                inputs[0] = inputs[0].unsqueeze(0)
                inputs[1] = inputs[1].unsqueeze(0)

            with torch.set_grad_enabled(False):
                rgb, t = inputs
                outputs, _ = self.model(rgb, t)
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
        dataloader.close()
        N = len(self.test_dataloader)
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        model_state_dic = self.model.state_dict()

        log_str = 'Test {}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                  'MSE {mse:.2f}, Time cost {time_cost:.1f}s'. \
            format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0],
                   time_cost=time.time() - epoch_start)
        logging.info(log_str)

        if game[0] < self.best_game[0]:
            self.best_mse = mse[0]
            self.best_game = game
            self.best_epoch = self.epoch
            logging.info(
                '*****Save Best GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                'MSE {mse:.2f} Model Epoch {e}'.format(
                    game0=self.best_game[0], game1=self.best_game[1],
                    game2=self.best_game[2], game3=self.best_game[3],
                    mse=self.best_mse, e=self.best_epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, "best_model.pth"))

        else:
            logging.info('Best GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                         'MSE {mse:.2f} Epoch {e}'.format(game0=self.best_game[0], game1=self.best_game[1],
                                                          game2=self.best_game[2], game3=self.best_game[3],
                                                          mse=self.best_mse, e=self.best_epoch))
