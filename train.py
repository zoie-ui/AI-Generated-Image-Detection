import math
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import transforms, datasets, utils
#import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os
import json
import time
import warnings
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from PIL import Image
from PIL import ImageFile

#from models.baseline.Zhang import HcNet
#from models.baseline.QuanNet import *
#from models.dual.DualCT import *
import argparse
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.utils import Logger
import pprint
from utils.loss import Loss_Function
import torchvision
from torchsummary import summary
from utils.MyDataset import *
import shutil


def parse_args():
    parse = argparse.ArgumentParser(description='XNet')
    # the path of inpainting images
   # parse.add_argument('-img_path', '--image_path', default= "/data/xiziyi/CG/SPL2018/raw/", type=str)
    # the path of masks
    #parse.add_argument('-msk_path', '--mask_path', default='/data/xiziyi/Inpainting/Mask_4w/', type=str)
    # the path of flist ./Flist/train.txt ./Flist/val.txt ./Flist/test.txt
    #parse.add_argument('-flist', default='./Flist/', type=str)
    # the path of save log and model params
    parse.add_argument('-c', '--checkpoints', default='/data/xiziyi/paper/base/resnet18/DA_256/')
    # the params of train
    parse.add_argument('-bs', '--batch_size', default=128, type=int, required=False)
    parse.add_argument('-e', '--epoch', default=120, type=int, required=False)
    # the learning rate
    parse.add_argument('-lr', default=2e-4, type=float, required=False)
    #parse.add_argument('-model', default=DualNet, required=False)
    parse.add_argument('-type', default="DALLE", required=False)
    parse.add_argument('-size', default=256, type=int, required=False)
    parse.add_argument('-split', default="split0", required=False)
    parse.add_argument('-decay', default=30, type=int, required=False)
    parse.add_argument('-decay_rate', default=0.1, type=float, required=False)
    #parse.add_argument('--size', default=128, type=int, required=False)
    # parse.add_argument('-optim',default='Adam',choices=['Adam','SGD'])
    args = parse.parse_args()
    return args


args = parse_args()
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if args.type == "DreamStudio":
    txt1 = './Flistds/' + args.split + '/'
elif args.type == 'DALLE':
    txt1 = './FlistDA/' + args.split + '/'
elif args.type == 'DsTok':
    txt1 = "./FlistDsTok/" + args.split + '/'
elif args.type == 'SPL2018':
    txt1 = "./FlistSPL2018/" + args.split + '/'

class Rgb2Gray(object):
    def __init__(self):
        '''rgb_image = object.astype(np.float32)
        if len(rgb_image.shape)!= 3 or rgb_image.shape[2] != 3:
            raise ValueError("input image is not a rgb image!")'''

    def __call__(self, img):
        L_image = img.convert('L')
        return L_image

data_transform = {
    "DALLE": transforms.Compose([
            transforms.ToTensor()
         ]),
    "DreamStudio": transforms.Compose([
            transforms.ToTensor()
         ]),
    "DsTok": transforms.Compose([
            transforms.CenterCrop(args.size),
            transforms.ToTensor()
        ]),
    "SPL2018": transforms.Compose([
            transforms.CenterCrop(args.size),
            transforms.ToTensor()
        ])
}

log_dir = os.path.join(args.checkpoints + 'tensorboard', 'train')
train_writer = SummaryWriter(log_dir=log_dir)
logger_test = Logger('test', args.checkpoints + 'test_log.log')

train_dataset = MyDataset(txt=txt1 + 'train.txt',
                          trans=data_transform[args.type],
                          type=args.type,
                          size=args.size)
train_num = len(train_dataset)
print(train_num)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=True)

validate_dataset = MyDataset(txt=txt1 + 'val.txt',
                             trans=data_transform[args.type],
                             type=args.type,
                             size=args.size)
val_num = len(validate_dataset)
print(val_num)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              drop_last=True)

# test
def train():
    # save_path /data/xiziyi/project/XNet/checkpoints/UNet/
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    logger = Logger('train', args.checkpoints + 'train_log.log')
    # logger1 = logging.getLogger('train')
    # writer = SummaryWriter(log_dir=os.path.join(args.checkpoints,'result-logs'))
    logger.info(pprint.pformat(args))
    logger.info(
        #f"model: {args.model} | "
        f"mkdir: {txt1} | "
        f"dataset: {args.type} | "
        f"input_size: {args.size} |\n"
    )
    net = models.resnet18(pretrained=False)
    net = net.cuda()
    net = nn.DataParallel(net)
    logger.info(net)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    #optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-4)
    # step_size: the time of involved
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay, gamma=args.decay_rate)
    logger.info(pprint.pformat(optimizer.state_dict()['param_groups'][0]))
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    # new_lr = 1e-4
    # sigmoid + BCE_loss
    best_loss = 10.0
    best_acc = 0.0
    loss_function = Loss_Function()
    for epoch in range(args.epoch):
        # if epoch % 20 == 0:
        #    new_lr = new_lr * 0.8
        #    optimizer = optim.Adam(net.parameters(), lr=new_lr)
        net.train()
        running_loss = 0.0
        running_corrects = 0.0
        time_start = time.perf_counter()
        warnings.filterwarnings('ignore')
        print('epoch[%d]  ' % epoch)
        step = 0

        for data in tqdm(train_loader, dynamic_ncols=True):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(images)
            preds = torch.max(outputs, dim=1)[1]
            # print(outputs.size())
            # print(labels.size())
            loss = loss_function(outputs, labels)
            # print("loss:",loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += (preds == labels).sum().item()
            step += 1
            # running_corrects += (preds == labels.cuda()).sum().item()

        scheduler.step()
        # writer.add_scalar('train/loss', running_loss/step, epoch)
        train_loss = running_loss / step
        train_acc = running_corrects / train_num
        logger.info(
            f"Train epoch {epoch}: "
            f" Time waste: {time.perf_counter() - time_start} |"
            # f"loss: {train_loss_meter['loss'].avg:.6f} |"
            f" loss: {train_loss:.6f} |"
            f" train_acc: {train_acc:.6f} |"
            # f"loss_pred: {train_loss_meter['loss_pred'].avg:.6f} |"
            f" lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f} |"
            # f"f1: |"
            # f"mIoU: |"
        )

        ########################################### validate ###########################################

        # net = torch.load('Result2.pth')
        net.eval()
        val_loss_sum = 0.0
        acc = 0.0
        with torch.no_grad():
            step = 0
            for val_data in tqdm(validate_loader):
                val_images, val_labels = val_data
                val_images = val_images.cuda()
                val_labels = val_labels.cuda()
                outputs = net(val_images)
                loss1 = loss_function(outputs, val_labels)
                result = torch.max(outputs, dim=1)[1]
                val_loss_sum += loss1.item()
                acc += (result == val_labels.cuda()).sum().item()
                step += 1
            val_loss = val_loss_sum / step
            val_acc = acc / val_num
            logger.info(
                f"Val epoch {epoch}: "
                f" loss: {val_loss:.3f} |"
                f" val_acc: {val_acc:.3f} |\n"
            )
            train_writer.add_scalars("loss", {"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc,
                                              "val_acc": val_acc}, epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                # best_loss = val_loss / val_num
                torch.save(
                    {'net': net.module.state_dict(),
                     'opt': optimizer.state_dict(),
                     'epoch': epoch,
                     'loss': val_loss,
                     'acc': best_acc}, args.checkpoints + 'model_best_acc.pt'
                )
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    {'net': net.module.state_dict(),
                     'opt': optimizer.state_dict(),
                     'epoch': epoch,
                     'loss': best_loss,
                     'acc': val_acc
                     }, args.checkpoints + 'model_best_loss.pt'
                )
    print('Finished Training')

def test(name, op=None):
    # print(image_path + '/' +
    # logger_test.info(summary(net,))
    test_dataset = MyDataset(txt=txt1 + 'test.txt',
                             trans=data_transform[args.type],
                             option=op,
                             type=args.type,
                             size=args.size)
    test_num = len(test_dataset)
    print(test_num)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              drop_last=False)
    net = models.resnet18(pretrained=False)
    '''
    dicts = torch.load(name)['net']
    mynet_dict = net.state_dict()
    new_dict = {k: v for k, v in dicts.items() if k in mynet_dict.keys()}
    mynet_dict.update(new_dict)
    net.load_state_dict(mynet_dict)
    '''
    #net.load_state_dict(torch.load(name)['net'])
    net.load_state_dict(torch.load(name)['net'])
    net = nn.DataParallel(net)
    net.cuda()
    print("model has been imported successfully!")

    net.eval()
    loss_function = nn.CrossEntropyLoss()
    test_loss = 0.0
    acc = 0.0
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    time_start = time.perf_counter()
    with torch.no_grad():
        for step, test_data in enumerate(test_loader, start=0):
            #print(test_data)
            test_images, test_labels = test_data
            test_images = test_images.cuda()
            test_labels = test_labels.cuda()
            outputs = net(test_images)
            b, _ = outputs.size()
            # bsx1xhxw
            result = torch.max(outputs, dim=1)[1]
            for i in range(b):
                if result[i] == test_labels[i] and test_labels[i] == 0:
                    TN += 1
                if result[i] == test_labels[i] and test_labels[i] == 1:
                    TP += 1
                if result[i] != test_labels[i]:
                    if result[i] == 1:
                        #target_pth = target1 + os.path.basename(img_path[i])
                        #shutil.copyfile(img_path[i], target_pth)
                        FP += 1
                    else:
                        #target_pth = target2 + os.path.basename(img_path[i])
                        #shutil.copyfile(img_path[i], target_pth)
                        FN += 1
            loss1 = loss_function(outputs, test_labels)
            test_loss += loss1.item()
        TPR = TP / (TP+FN)
        TNR = TN / (TN+FP)
        test_acc = (TP+TN) / test_num
        logger_test.info(
            f"[dataset {args.type}]: {args.size}x{args.size} "
            f"time_waste:{time.perf_counter() - time_start:.3f} "
            f"test: {op} |"
            f"TPR: {TPR:.4f} |"
            f"TNR: {TNR:.4f} |"
            f"loss: {(test_loss / step):.4f} |"
            f"acc: {test_acc:.4f} |"
        )


if __name__ == '__main__':
    #bacc = args.checkpoints + 'DsTok.pth'
    train()
    bacc = args.checkpoints + 'model_best_acc.pt'
    bloss = args.checkpoints + 'model_best_loss.pt'
    #test(bacc)
    #test(bloss)
    #test(bacc, "rotate")
    #test(bloss, "rotate")
    #options = ["gb3x3", "gb5x5", "median3x3", "median5x5", "avg3x3", "avg5x5", "resize0.5", "resize0.8", "test_75", "test_85", "test_95"]
    options = ['color', 'brightness', 'contrast', 'sharp', "rotate", "gb5x5", "avg5x5"]
    #options = ["gb3x3", "gb5x5"]
    for op in options:
        test(bacc, op)

    #for op in options:
    #    test(bloss, op)


