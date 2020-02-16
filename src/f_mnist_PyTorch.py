import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ignite.engine 
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

import cv2
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import argparse

from keras.datasets import fashion_mnist
import os
from PIL import Image

#class Autoencoder(chainer.Chain):
#    def __init__(self):
#        super(Autoencoder, self).__init__(
#            encoder = L.Linear(784, 256),
#            decoder = L.Linear(256, 784))

#    def __call__(self, x, hidden=False):
#        h = F.relu(self.encoder(x))
#        if hidden:
#            return h
#        else:
#            return F.relu(self.decoder(h))

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 784)

    def forward(self,x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return F.log_softmax(x, dim=-1)

class make_dir(object):
    def __init__(self,folder_path):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

class show_img_data(object):
    def __init__(self, file_path, plot_idx, shape_size):
        self.out_file = file_path
        self.plot_idx = plot_idx
        self.shape_size = shape_size
    
    def make_data_nonmodel(self, data, idx, num):
        data_list = []
        for i in range(0 , num):
            data_list.append((data[i], idx[i]))
        
        return data_list
    
    def make_data_usemodel(self, in_data, idx, num, model):
        data_list = []
        for i in range(0 , num):
            print(idx[i])
            pred_data = model.predictor(np.array([in_data[i]]).astype(np.float32)).data
            data_list.append((pred_data, idx[i]))
        
        return data_list
    
    def plot_mnist_data(self,data,epoch):
        for index, (data, label) in enumerate(data):
            plt.subplot(self.plot_idx, self.plot_idx, index + 1)
            plt.axis('off')
            plt.imshow(data.reshape(self.shape_size, self.shape_size), cmap=cm.gray_r, interpolation='nearest')
            n = int(label)
            plt.title(n, color='red')
        
        make_dir(self.out_file)
        plt.savefig("./{0}/size_{1}_epoch_{2}.png".format(self.out_file, self.shape_size, epoch))
        # plt.show()

class Makedataset(Dataset):
    def __init__(self, inputdata, outputdata, transform=None):
        self.transform = transform
        self._input = inputdata
        self._output = outputdata

    def __len__(self):
        return len(self._input)

    def __getitem__(self, idx):
        if self.transform:
            in_data = self.transform(self._input[idx])
            #out_data = self.transform(self._output[idx])
        else:
            in_data = self._input[idx]
            out_data =  self._output[idx]
        
        return in_data, out_data

def training_data(args, x_train, y_train, x_test, y_test):
    PLOT_NUMS = 3
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    #plot_input = show_img_data(args.out,3,28)
    #p_data = plot_input.make_data_nonmodel(plot_data, plot_idx, 9)
    #plot_input.plot_mnist_data(p_data, 0)

    # 学習データの作成
    # train_in = [cv2.resize(img, dsize=(14,14)) for img in x_train]
    # test_data = [cv2.resize(img, dsize=(14,14)) for img in plot_data]
    #x_train = [np.ravel(i) for i in x_train]
    #x_test = [np.ravel(i) for i in x_test]
    
    #x_train = [i for i in x_train]
    #x_test = [i for i in x_test]

    #train_in_data = np.array([np.ravel(data) for data in x_train])
    #test_in_data = np.array([np.ravel(data) for data in x_test])

    #train_in_data = torch.Tensor(train_in_data)
    #train_out_data = torch.FloatTensor(train_in_data)
    #test_in_data = torch.Tensor(test_in_data)
    #y_train = torch.LongTensor(y_train)
    #y_test = torch.LongTensor(y_test)

    plot_data = x_test[0:PLOT_NUMS**2]
    plot_idx = y_test[0:PLOT_NUMS**2]

    #train_dataset = Makedataset(train_in_data, train_in_data)
    #test_dataset = Makedataset(test_in_data, test_in_data)
    #train_dataset = TensorDataset(train_in_data, train_out_data)
    #test_dataset = TensorDataset(test_in_data, test_in_data)

    #train_dataset=torch.from_numpy(train_dataset)
    #train_dataset=train_dataset.to(torch.float)
    #train_dataset = torch.tensor(train_dataset, dtype=torch.float32, device="cpu")
    #test_dataset = torch.tensor(test_dataset, dtype=torch.float32, device="cpu")

    #print(train_dataset[0])
    #a = Makedataset(train_in_data, train_in_data)
    #print(len(a[0][0]))

    #train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize ,True ,True)
    #test_iter = chainer.iterators.SerialIterator(test_dataset, len(test_dataset), False, False)

    datasets = Makedataset(x_train, x_train)

    print(len(datasets))
    print(datasets[0])
    print(datasets)

    train_iter = DataLoader(Makedataset(x_train, x_train), args.batchsize, shuffle=True)
    test_iter = DataLoader(Makedataset(x_test, x_test), batch_size=len(x_test), shuffle=False)
    
    #model = L.Classifier(Autoencoder(), lossfun=F.mean_squared_error)
    # モデルの定義
    model = Autoencoder()
    
    #Loss関数の指定
    criterion = nn.CrossEntropyLoss()

    #model.compute_accuracy = False
    #optimizer = chainer.optimizers.Adam(0.01)
    #optimizer.setup(model)

    # optimizerの設定
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    #trainer = training.Trainer(updater, (args.epoch, 'epoch'), out="result")
    #trainer.extend(extensions.LogReport())
    #trainer.extend(extensions.PrintReport( ['epoch', 'main/loss']))
    #trainer.extend(extensions.ProgressBar())

    trainer = ignite.engine.create_supervised_trainer(model,optimizer,criterion,device='cpu')
    evaluator = ignite.engine.create_supervised_evaluator(
        model,
        metrics={
            'accuracy': ignite.metrics.Accuracy(),
            'loss': ignite.metrics.Loss(criterion),
        },
        device='cpu')
    
    trainer.run(train_iter, max_epochs=10)

    #trainer.run()
    ##model.to_cpu()

    #plot_out = show_img_data(args.out,PLOT_NUMS,28)
    #outp_data = plot_out.make_data_usemodel(plot_data, plot_idx, PLOT_NUMS**2 , model)
    #plot_out.plot_mnist_data(outp_data, 100)
    

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--batchsize", "-b", type=int, default=128, help="Batchsize")
    ap.add_argument("--epoch", "-e", type=int, default=100, help="Number of epochs")
    ap.add_argument("--out", "-o", type=str, default="results", help="Output directory name")
    ap.add_argument("--gpu", "-g", type=int, default=-1, help="GPU ID")
    ap.add_argument("--lr", "-l", type=float, default=0.01, help="Learning rate of SGD")

    args = ap.parse_args()
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') # int型をfloat32型に変換
    x_test = x_test.astype('float32') # int型をfloat32型に変換
    t_train = y_train.astype('int32') # 一応int型に
    t_test = y_test.astype('int32') # 一応int型に
    x_train /= 255 # [0-255]の値を[0.0-1.0]に変換
    x_test /= 255 # [0-255]の値を[0.0-1.0]に変換

    x_train = [i for i in x_train]
    x_test = [i for i in x_test]

    train_in_data = np.array([np.ravel(data) for data in x_train])
    test_in_data = np.array([np.ravel(data) for data in x_test])

    #train_in_data = torchvision.transforms.ToTensor()(train_in_data)
    #test_in_data = torchvision.transforms.ToTensor()(test_in_data)

    #print(len(train_in_data[0][0]))

    train_in_data = torch.from_numpy(train_in_data)
    test_in_data = torch.from_numpy(test_in_data)

    #train_in_data = torch.tensor(train_in_data, dtype = torch.long, device = 'cpu')
    #test_in_data = torch.tensor(test_in_data, dtype = torch.long, device = 'cpu')

    training_data(args, train_in_data, t_train, test_in_data, t_test)

if __name__ == "__main__":
    main()




