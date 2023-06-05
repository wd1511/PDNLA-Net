import Network
from colorconvert import *
import math
import torch
from torchvision import utils
import os
import argparse
from torch import nn
from PIL import Image, ImageFont, ImageDraw
import numpy as np

def image_processing(imgpath_gray, imgpath_color, device):
    img1 = []
    img2 = []

    img = Image.open(imgpath_gray).convert("RGB").resize((512, 512))
    img_a = (
        torch.from_numpy(np.array(img))
        .to(torch.float32)
        .div(255)
        .add_(-0.5)
        .mul_(2)
        .permute(2, 0, 1)
    )

    img = Image.open(imgpath_color).convert("RGB").resize((512, 512))
    img_b = (
        torch.from_numpy(np.array(img))
        .to(torch.float32)
        .div(255)
        .add_(-0.5)
        .mul_(2)
        .permute(2, 0, 1)
    )
    img1.append(img_a)
    img2.append(img_b)

    img1 = torch.stack(img1, 0).to(device)
    img2 = torch.stack(img2, 0).to(device)
    return img1, img2

def image_result(image_in, H, image_dir):
    image_list = []
    image_list.append(image_in)
    b, c, h, w = image_list[0].size()
    W = math.ceil(len(image_list)/H)
    flag = 0
    image_out_list = []
    for i in range(W):
        x = torch.FloatTensor(H, c, h, w)
        for j in range(H):
            if flag < len(image_list):
                img_this = image_list[flag]
                x[j, :, :, :] = img_this[0, :, :, :].to("cpu")
            flag = flag + 1
        image_out_list.append(x)
    imgsets1 = torch.cat(image_out_list, 0)
    grid = utils.save_image(
        imgsets1, image_dir, nrow=H, normalize=True, range=(-1, 1)
    )

parser = argparse.ArgumentParser()
parser.add_argument("--files1", type=str,default="./test_image/contents")
parser.add_argument("--files2", type=str, default="./test_image/color")
parser.add_argument("--ckpt_dir", type=str, default="./ckpts/model1_ema.pt")
parser.add_argument("--result_dir", type=str, default="./results/")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

ckpt = torch.load(args.ckpt_dir, map_location=lambda storage, loc: storage)
ckpt_args = ckpt["args"]
device = args.device

myNet = Network.PDNLAnet(ckpt_args.channel).to(device)
myNet.load_state_dict(ckpt["PDNLA_ema"])
myNet.eval()


img_filename_gray = os.listdir(args.files1)
img_filename_color = os.listdir(args.files2)
img_filename_gray.sort()
img_filename_color.sort()

for i in range(len(img_filename_gray)):
    imgpath_gray = os.path.join(args.files1, img_filename_gray[i])
    imgpath_color = os.path.join(args.files2, img_filename_color[i])
    print(imgpath_gray)
        
    img1, img2 = image_processing(imgpath_gray, imgpath_color, device)
    img3 = myNet(img1,img2)
    img4 = gray_replace(img1,img3)
    out_filename = os.path.join(args.result_dir, img_filename_gray[i])

    image_result(img4, 1, out_filename)