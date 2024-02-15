import os
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import soundfile as sf

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from data import RWAVSDataset
from trainer import Trainer
from model import ANeRF


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log-dir', type=str, default="logs/")
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--result-dir', type=str, default="results/")
    parser.add_argument('--resume-path', type=str)
    parser.add_argument('--best-ckpt', type=str)
    
    # dataset
    parser.add_argument('--data-root', type=str, default="")
    parser.add_argument('--no-position', action="store_true")
    parser.add_argument('--no-orientation', action="store_true")

    # model
    parser.add_argument('--conv', action="store_true")
    parser.add_argument('--p', type=float, default=0)
    parser.add_argument('--wave', action="store_true")
    parser.add_argument('--energy', action="store_true")

    # train
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-epoch', type=int, default=100)

    # eval
    parser.add_argument('--eval', action="store_true")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.benchmark = True
    device = "cuda:0"

    print(f"Use convolution: {args.conv}")
    print(f"Use position: {not args.no_position}")
    print(f"Use orientation: {not args.no_orientation}")
    print(f"Use energy loss: {args.energy}")
    print(f"Use wave loss: {args.wave}")

    model = ANeRF(conv=args.conv, p=args.p)
    model.to(device)

    if args.eval:
        val_dataset = RWAVSDataset(args.data_root, "val", no_pos=args.no_position, no_ori=args.no_orientation)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, pin_memory=True)

        ckpt_path = os.path.join(args.log_dir, args.output_dir if args.output_dir else f"{args.batch_size}_{args.lr}/", f"{99}.pth")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"load parameters from {ckpt_path}")
        del checkpoint
        trainer = Trainer(
            args,
            model,
            criterion=None,
            optimizer=None,
            log_dir=args.log_dir,
            last_epoch=-1,
            device=device,
        )
        _, prd = trainer.eval(val_dataloader, save=True)
        pickle.dump(prd, open(os.path.join(args.log_dir, args.output_dir, "val_vis.pkl"), "wb"))

    else:
        train_dataset = RWAVSDataset(args.data_root, "train", no_pos=args.no_position, no_ori=args.no_orientation)
        val_dataset = RWAVSDataset(args.data_root, "val", no_pos=args.no_position, no_ori=args.no_orientation)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, pin_memory=True)

        if args.resume_path:
            checkpoint = torch.load(args.resume_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            print(f"load parameters from {args.resume_path}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.resume_path:
            optimizer.load_state_dict(checkpoint["optimizer"])
        trainer = Trainer(
            args,
            model,
            criterion=None,
            optimizer=optimizer,
            log_dir=args.log_dir,
            last_epoch=checkpoint["epoch"] if args.resume_path else -1,
            last_iter=checkpoint["iter"] if args.resume_path else -1,
            device=device,
        )
        st_epoch = checkpoint["epoch"] + 1 if args.resume_path else 0
        ed_epoch = args.max_epoch
        t = tqdm(total=ed_epoch-st_epoch, desc="[EPOCH]")
        eval_loss_list = []
        if args.resume_path:
            del checkpoint
        for epoch in range(st_epoch, ed_epoch):
            trainer.train(train_dataloader)
            eval_loss = trainer.eval(val_dataloader)
            eval_loss_list.append(eval_loss)
            trainer.save_ckpt()
            trainer.epoch += 1
            t.update()
        t.close()
        pickle.dump(eval_loss_list, open(os.path.join(args.log_dir, args.output_dir, "eval_loss.pkl"), "wb"))

if __name__=='__main__':
    main()