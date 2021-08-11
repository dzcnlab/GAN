import pathlib
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import os
import os
from torchvision import datasets, transforms
# import cv2
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
# from keras.datasets import mnist
from torchvision.utils import save_image
from astropy.io import fits
import numpy as np
import glob
import torch.utils.data as Data
import matplotlib.pyplot as plt
import random
import warnings
import itertools
import logging
import time # 时间
import os # 路径
from logging import handlers
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射
    
    def __init__(self,filename,level='info'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter()#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = logging.FileHandler(filename=filename, mode='w', encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        # th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

class ImagesFolder(Dataset):
    def __init__(self, root, transform=None,distributed=False,open_mode=None):
        self.images_path = self.getImagesPath(root,distributed)
        self.transform=transform
        self.open_mode = open_mode

    def getImagesPath(self,root,distributed=False):
        if distributed:
            images_path=list(pathlib.Path(root).rglob('*.png'))
        else:
            images_path = list(pathlib.Path(root).glob('*.png'))
        return images_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        path = self.images_path[index]
        image = Image.open(path)
        if self.open_mode is not None: image = image.convert(self.open_mode)
        if self.transform is not None: image = self.transform(image)
        return image,0

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256*256)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True) # inplace设为True，让操作在原地进行
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.MaxPool2d(stride=1, kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.MaxPool2d(stride=1, kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2,padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.MaxPool2d(stride=1, kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # nn.MaxPool2d(stride=1, kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.MaxPool2d(stride=1, kernel_size=2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 1, 3, stride=1, padding=1),
            nn.Tanh(),
            # nn.MaxPool2d(stride=1, kernel_size=2)
        )

    def forward(self, x):
        x = self.fc1(x)
        print('fc1:', x.shape)
        x = x.view(-1, 1, 256, 256)
        x = self.br(x)
        print('br:', x.shape)
        x = self.conv1(x)
        print('conv1:', x.shape)
        x = self.conv2(x)
        print('conv2:', x.shape)
        x = self.conv3(x)
        print('conv3:', x.shape)
        x = self.conv4(x)
        print('conv4:', x.shape)
        x = self.conv5(x)
        print('conv5:', x.shape)
        output = self.conv6(x)
        print('G_output_shape:',output.shape)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.pl1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.pl2 = nn.AvgPool2d(2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 8 * 8, 1024),
            nn.LeakyReLU(0.2,True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        print('conv1:', x.shape)
        x = self.pl1(x)
        print('pl1:', x.shape)
        x = self.conv2(x)
        print('conv2:', x.shape)
        x = self.pl2(x)
        print('pl2:', x.shape)
        x = x.view(x.shape[0], -1)
        print('view:', x.shape)
        x = self.fc1(x)
        print('fc1:', x.shape)
        output = self.fc2(x)
        print('D_output_shape:',output.shape)
        return output

def G_train(input_dim):
    G_optimizer.zero_grad()

    noise = torch.randn(batch_size, input_dim).to(device='cuda:0')
    real_label = torch.ones(batch_size).to(device='cuda:0')
    fake_img = G(noise)
    D_output = D(fake_img)
    G_loss = criterion(D_output, real_label)

    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

def D_train(real_img, input_dim):
    D_optimizer.zero_grad()

    real_label = torch.ones(real_img.shape[0]).to(device='cuda:0')
    D_output = D(real_img)
    D_real_loss = criterion(D_output, real_label)

    noise = torch.randn(batch_size, input_dim, requires_grad=False).to(device='cuda:0')
    fake_label = torch.zeros(batch_size).to(device='cuda:0')
    fake_img = G(noise)
    D_output = D(fake_img)
    D_fake_loss = criterion(D_output, fake_label)

    D_loss = D_real_loss + D_fake_loss

    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()

def save_img(img, img_name):
    img = 0.5 * (img + 1)
    img = img.clamp(0, 1)
    save_image(img, "./imgs/" + img_name)
    # print("image has saved.")

if __name__ == '__main__':
    device = torch.device('cuda:0')
    cuda = True if torch.cuda.is_available() else False
    batch_size = 10
    epoch_num = 200
    lr = 0.0002
    input_dim = 2000
    input_dim = np.array(input_dim)
    input_dim = torch.from_numpy(input_dim)
    input_dim = input_dim.to(device = device)
    log = Logger('SGD.log', level='info')
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

    if not os.path.exists("./imgs"):
        os.makedirs("./imgs")

    # 调整图片大小，转化为张量，调整值域为-1到1
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载数据集
    train_ds=ImagesFolder(r'/mnt/storage-ssd/tanlei/work/pytorch/CNN/DZ/dz/x/', transform)
    # 观察数据集
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import numpy as np

    train_dl = DataLoader(train_ds, batch_size=batch_size)
    # for images, labels in train_dl:
    #     images = make_grid(images, nrow=4)
    #     images = np.transpose(images.data * 0.5 + 0.5, [1, 2, 0])
    #     plt.imshow(images)
    #     plt.show()
    
    G = Generator(input_dim)
    D = Discriminator()

    # 指明损失函数和优化器
    criterion = nn.BCELoss()
    if cuda:
        G.cuda()
        D.cuda()
        criterion.cuda()

    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    fake_no = torch.randn(128, input_dim)
    # print(G)
    # print(D)
    for epoch in range(1, epoch_num + 1):
        log.logger.info("epoch: %d"%(epoch))
        for batch, (x, _) in enumerate(train_dl):
            # 对判别器和生成器分别进行训练，注意顺序不能反
            # print(type(x), type(input_dim))
            x = x.to(device = device)
            D_loss=D_train(x, input_dim)
            G_loss=G_train(input_dim)
            # x_real=np.array(x)
            if epoch == 1:
                save_img(x[0], "img_real.png")
            if batch % 5 == 0:
                log.logger.info("[ %d / %d ]  g_loss: %.6f  d_loss: %.6f" % (batch, 20, float(G_loss), float(D_loss)))

            if batch % 5 == 0:
                Gs = Generator(input_dim)
                Gs.to(device='cpu')
                fake_img = Gs(fake_no)
                save_img(fake_img[0], "img_" + str(epoch) + "_" + str(batch) + ".png")