import math
import os
from collections import OrderedDict

import numpy as np
import torch
import time
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
from PIL import Image, ImageTk
from matplotlib import cm

def loadFromFile(path, img_width=300, img_height=300, mode=Image.ANTIALIAS):
    try:
        img_data = Image.open(path)
        img_data = img_data.convert('RGB')
        img_data_resize = img_data.resize(
            (img_width, img_height), mode)
        img_tk = ImageTk.PhotoImage(image=img_data_resize)
        # img_tk.paste(img_data_resize)
        return img_tk, img_data
    except:
        return None, None

def loadFromNumpy(img_np, img_width=300, img_height=300,  mode=Image.NEAREST, scale=True, cmap=True):
    if scale:
        img_np = img_np*255
    try:
        img_data = Image.fromarray(np.uint8(img_np), 'RGB')
        if cmap:
            img_data_convert = Image.fromarray(cm.jet(img_np[...,0]/255, bytes=True))
        else:
            img_data_convert = img_data
        img_data_resize = img_data_convert.resize((img_width, img_height), mode)
        img_tk = ImageTk.PhotoImage(image=img_data_resize)
        # img_tk.paste(img_data_resize)
        return img_tk, img_data
    except:
        return None, None


def tensor2Grid(val):
    scale_each = True
    t_0 = time.time()
    if val.dim() == 2:
        val = val.unsqueeze(2).unsqueeze(2)
        scale_each = False

    img = tv.utils.make_grid(val.transpose(
        0, 1),
        nrow=int(math.sqrt(val.size(1))),
        normalize=True,
        scale_each=scale_each
    )
    # t_1 = time.time()

    npimg = img.cpu().detach().numpy() if img.is_cuda else img.detach().numpy()
    # t_2 = time.time()
    npimg = np.transpose(npimg, (1, 2, 0))
    # t_3 = time.time()

    # print("t1:", t_1-t_0)
    # print("t2:", t_2-t_1)
    # print("t3:", t_3-t_2)

    return npimg
