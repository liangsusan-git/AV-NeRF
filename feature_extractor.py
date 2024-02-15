import os
import PIL
import glob
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import soundfile as sf

import torch
from torchvision.models import resnet18, ResNet18_Weights

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default="")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--save-dir', type=str, default="")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    weights = ResNet18_Weights.DEFAULT
    original_resnet = resnet18(weights=weights, progress=False).eval()
    layers = list(original_resnet.children())[:-1]
    model = torch.nn.Sequential(*layers)
    model.to("cuda:0")
    transforms = weights.transforms()

    rgb_list = sorted(glob.glob(os.path.join(args.data_dir, args.split, "rgb/*.png")))
    depth_list = sorted(glob.glob(os.path.join(args.data_dir, args.split, "depth/*.png")))
    features = {"rgb": [],
                "depth": []}
    for rgb in rgb_list:
        rgb = PIL.Image.open(rgb).convert('RGB')
        rgb = transforms(rgb).unsqueeze(0) # [1, 3, h, w]
        with torch.no_grad(): feature = model(rgb.to("cuda:0")).squeeze().cpu().numpy()
        features["rgb"].append(feature)
    for depth in depth_list:
        depth = PIL.Image.open(depth).convert('RGB')
        depth = transforms(depth).unsqueeze(0) # [1, 3, h, w]
        with torch.no_grad(): feature = model(depth.to("cuda:0")).squeeze().cpu().numpy()
        features["depth"].append(feature)
    pickle.dump(features, open(os.path.join(args.save_dir, f"feats_{args.split}.pkl"), "wb"))