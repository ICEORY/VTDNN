import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
from PIL import Image, ImageTk

import graph
import model as md
import utils


class DeepModel(object):
    def __init__(self, net_type="LeNet5", weight_file=None, gpu=False):
        self.model = None
        self.layer_feature = OrderedDict()
        self.layer_mask = OrderedDict()
        self.layer_count = np.zeros(3)  # number of layers: conv, relu, fc
        self.model_list = [
            "LeNet5", 
            "ResNet-18", 
            "ResNet-50", 
            # "LCP-ResNet-18",
            "VGG-16"]

        self.net_type = net_type
        self.gpu = gpu
        self.labels = None
        self.dataset = None
        if net_type == "LeNet5":
            self.model = md.LeNet5()
            self.loadWeight(weight_file="./pretrained/LeNet5_Baseline.pkl")
            self.dataset = "MNIST"

        elif net_type in ["ResNet-18", "ResNet-50", "VGG-16", "LCP-ResNet-18"]:
            if net_type in ["ResNet-18", "LCP-ResNet-18"]:
                self.model = md.ResNet(depth=18)
                if net_type == "ResNet-18":
                    self.loadWeight(weight_file="./pretrained/ResNet18_Baseline.pth")
                else:
                    self.loadWeight(weight_file="./pretrained/LCP-ResNet-18-0.5.pth")
            elif net_type == "ResNet-50":
                self.model = md.ResNet(depth=50)
                self.loadWeight(weight_file="./pretrained/ResNet50_Baseline.pth")
            elif net_type == "VGG-16":
                self.model = md.VGG(depth=16)
                self.loadWeight(weight_file="./pretrained/VGG16_Baseline.pth")
            self.dataset = "ImageNet"

        if self.dataset == "MNIST":
            self.input_size = 28
            self.input_scale = 28
            self.norm = ([0.1307], [0.3081])
            self.test_input = torch.zeros(1, 1, 28, 28, requires_grad=True)
        elif self.dataset == "ImageNet":
            self.input_scale = 256
            self.input_size = 224
            self.norm = ([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
            self.test_input = torch.zeros(1, 3, 224, 224, requires_grad=True)
            self.labels = utils.loadLabels(file_path="./labels/ImageNet.txt")
            

        if self.model is not None:
            self.replaceLayer()
            self.model.eval()
            if gpu:
                self.model.cuda()
            for layer in self.model.modules():
                if isinstance(layer, (md.MaskConv2d, nn.ReLU, nn.Linear)):  # remove cnn, nn.Conv2d,
                    layer.register_forward_hook(self.forwardHook)
                """
                if isinstance(layer, nn.Conv2d):
                    self.layer_count[0] += 1
                    key = "CONV_%d" % (self.layer_count[0])
                    self.layer_feature[key] = None
                el"""
                if isinstance(layer, md.MaskConv2d):
                    self.layer_count[0] += 1
                    key = "PrConv_%d" % (self.layer_count[0])
                    self.layer_feature[key] = None
                elif isinstance(layer, nn.ReLU):
                    self.layer_count[1] += 1
                    key = "ReLU_%d" % (self.layer_count[1])
                    self.layer_feature[key] = None
                elif isinstance(layer, nn.Linear):
                    self.layer_count[2] += 1
                    key = "FC_%d" % (self.layer_count[2])
                    self.layer_feature[key] = None
            # self.visualNet()
    
    def replaceLayer(self):
        if self.net_type in ["LCP-ResNet-18"]:
            for module in self.model.modules():
                if isinstance(module, (md.BasicBlock)):
                    # replace conv2
                    temp_conv = md.MaskConv2d(
                        in_channels=module.conv2.in_channels,
                        out_channels=module.conv2.out_channels,
                        kernel_size=module.conv2.kernel_size,
                        stride=module.conv2.stride,
                        padding=module.conv2.padding,
                        bias=(module.conv2.bias is not None))

                    temp_conv.weight.data.copy_(module.conv2.weight.data)
                    if module.conv2.bias is not None:
                        temp_conv.bias.data.copy_(module.conv2.bias.data)
                    module.conv2 = temp_conv

    def loadWeight(self, weight_file):
        # try:
        if weight_file is not None and os.path.isfile(weight_file):
            state_dict = torch.load(weight_file)
            if self.net_type == "LCP-ResNet-18":
                state_dict = state_dict["model"]
            if self.gpu:
                self.model = self.model.cpu()
                self.model.load_state_dict(state_dict)
                self.model.cuda()
            else:
                self.model.load_state_dict(state_dict)
            # print("load weight succeed")
        # except:
        #    print("failed to load weight")

    def imgPreprocess(self, image):
        if self.net_type == "LeNet5":
            image = image.convert("L")
            # print(type(image))
        else:
            image = image.convert("RGB")
        transformer = transforms.Compose([
            transforms.Resize(self.input_scale),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm[0], std=self.norm[1])
        ])

        output = transformer(image)
        if output.dim() == 3:
            return output.unsqueeze(0)
        elif output.dim() == 2:
            return output.unsqueeze(0).unsqueeze(1)
        elif output.dim() == 4:
            return output
        else:
            return None

    def forwardHook(self, module, input, output):
        """
        if isinstance(module, nn.Conv2d):
            self.layer_count[0] += 1
            key = "CONV_%d" % (self.layer_count[0])
            self.layer_feature[key] = output.clone()
            # print(key)
        el"""
        if isinstance(module, md.MaskConv2d):
            self.layer_count[0] += 1
            key = "PrConv_%d" % (self.layer_count[0])
            self.layer_feature[key] = input[0].clone()
            self.layer_mask[key] = module.d.clone()
        elif isinstance(module, nn.ReLU):
            self.layer_count[1] += 1
            key = "ReLU_%d" % (self.layer_count[1])
            self.layer_feature[key] = output.clone()
            # print(key)
        elif isinstance(module, nn.Linear):
            self.layer_count[2] += 1
            key = "FC_%d" % (self.layer_count[2])
            self.layer_feature[key] = output.clone()
            # print(key)

    def featureProcess(self):
        if self.layer_feature is not None:
            for key, val in self.layer_feature.items():
                print("val size: ", val.size())
                scale_each = True
                if val.dim() == 2:
                    val = val.unsqueeze(2).unsqueeze(2)
                    scale_each = False
                img = tv.utils.make_grid(val.transpose(
                    0, 1),
                    nrow=int(math.sqrt(val.size(1))),
                    normalize=True,
                    scale_each=scale_each
                )
                print("img size: ", img.size())
                npimg = img.cpu().detach().numpy() if self.gpu else img.detach().numpy()
                npimg = np.transpose(npimg, (1, 2, 0))
                self.layer_feature[key] = npimg

    def forward(self, image):
        self.layer_feature = OrderedDict()
        self.layer_mask = OrderedDict()
        self.layer_count.fill(0)
        image = self.imgPreprocess(image)
        if self.gpu:
            image = image.cuda()
        # print(image)
        # print(image.size())
        output = self.model(image).squeeze()
        output = output.squeeze()
        output = output.cpu().detach().numpy() if self.gpu else output.detach().numpy()
        # self.featureProcess()
        max_predict = output.argmax()
        # print(max_predict, output[max_predict])
        if self.dataset == "MNIST":
            msg_str = "Number: %d" % max_predict
        elif self.dataset == "ImageNet":
            if self.labels is not None:
                msg_str = "ID: %d, Name: %s"%(max_predict, self.labels[max_predict])
            else:
                msg_str = "ID: %d" % (max_predict)
        else:
            msg_str = "Predict: %d" % max_predict
                
        return msg_str

    def visualNet(self):
        path = ".\\src\\temp_net.png"
        if self.test_input is not None:
            if self.gpu:
                self.test_input = self.test_input.cuda()
            
            self.layer_feature = OrderedDict()
            self.layer_count.fill(0)
            output = self.model(self.test_input)
            gr = graph.Graph()
            gr.draw(output)
            # gr.save(file_name=path)
        return path
