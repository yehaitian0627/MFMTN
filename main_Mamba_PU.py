from email.policy import default

import torch
from models import videomamba

import os
from config_PU import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time

from utils.utils import Logger, mkdirs
import get_cls_map_PU

def loadData():
    if config.data == 'Indian':
        data = sio.loadmat('./data/Indian/Indian_pines.mat')['indian_pines_corrected']
        labels = sio.loadmat('./data/Indian/Indian_pines_gt.mat')['indian_pines_gt']

    return data, labels

def applyPCA(X, numComponents):
    # Principal component analysis on HSI data
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

def createImageCubes(X, y, windowSize, removeZeroLabels = True):

    margin = windowSize // 2
    H, W, C = X.shape

    # 补零处理（用于提取边缘patch）
    padded_X = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

    patchesData = []
    patchesLabels = []

    for r in range(margin, H + margin):
        for c in range(margin, W + margin):
            label = y[r - margin, c - margin]
            if removeZeroLabels and label == 0:
                continue  # 跳过标签为0的像素
            patch = padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1, :]
            patchesData.append(patch)
            patchesLabels.append(label)

    patchesData = np.array(patchesData, dtype=np.float32)
    patchesLabels = np.array(patchesLabels, dtype=np.int64)

    if removeZeroLabels:
        patchesLabels -= 1  # 标签从 1-based 转为 0-based

    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testRatio,random_state=randomState,stratify=y)

    return X_train, X_test, y_train, y_test

def create_data_loader():

    X, y = loadData()
    index = np.nonzero(y.reshape(y.shape[0]*y.shape[1]))
    index = index[0]
    # The proportion of test samples
    test_ratio = config.test_ratio
    # patch size
    patch_size = config.patch_size
    # The dimension after PCA
    pca_components = config.pca_components

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)
    groundtruth = y

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)
    print('\n... ... create data cubes ... ...')
    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # Create train loader and test loader
    X = TestDS(X, y)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=config.BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               drop_last=True
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=config.BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=False
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                               batch_size=config.BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=False
                                             )

    return train_loader, test_loader, y, index, all_data_loader, groundtruth

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]
    def __len__(self):

        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len

def train(train_loader):
    net = videomamba.VisionMamba(
        model_type=config.model_type,
        embed_dim=config.embed_dim,
        d_state=config.d_state,  # 256,  # State dimension of the model
        ssm_ratio = config.ssm_ratio,
        num_classes=config.num_classes,  # Number of output classes
        depth=config.depth,
        pos=config.pos,
        cls=config.cls,
        conv3D_channel=config.conv3D_channel,
        conv3D_kernel_1=config.conv3D_kernel_1,
        conv3D_kernel_2=config.conv3D_kernel_2,
        conv3D_kernel_3=config.conv3D_kernel_3,
        dim_patch=config.dim_patch,
        dim_linear_1=config.dim_linear_1,
        dim_linear_2=config.dim_linear_2,
        dim_linear_3=config.dim_linear_3,
    ).cuda()

    total = sum([param.nelement() for param in net.parameters()])
    print(total)

    # Use cross entropyloss function
    criterion = nn.CrossEntropyLoss()
    # Initializes the optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # Start training
    total_loss = 0
    for epoch in range(config.train_epoch):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()  #[64, 1, 30, 15, 15]
            outputs, _ = net(data)   #[64, 9]
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log.write('[Epoch: %d]   [loss avg: %.4f] \n' % (epoch + 1, total_loss / (epoch + 1)))

    log.write('Finished Training')
    from thop import profile
    flops, params = profile(net, inputs=(data[0].unsqueeze(dim=0),))
    print('Params = ' + format(str(params / 1000 ** 2), '.6') + 'M')
    print('FLOPs = ' + format(str(flops / 1000 ** 3), '.6') + 'G')
    return net, flops, params

def mytest(net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        outputs, features = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        features = features.detach().cpu().numpy()
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            y_feature = features
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
            y_feature = np.concatenate((y_feature, features))

    return y_pred_test, y_test, y_feature

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':
    mkdirs(config.checkpoint_path, config.checkpoint_path, config.logs)
    log = Logger()
    log.open(config.logs + config.data + '_log.txt', mode='a')

    oa = []
    acc = []
    aa = []
    kappa = []
    for num in range(config.test_epoch):
        train_loader, test_loader, y_all, index, all_data_loader, y = create_data_loader()
        tic1 = time.perf_counter()
        # net, flops, params = train(train_loader)
        net, flops, params = train(train_loader)
        toc1 = time.perf_counter()
        print("训练时间: {:.2f} 秒".format(toc1 - tic1))
        # Save model parameters
        # torch.save(net.state_dict(), 'LSFAT_params.pth')
        tic2 = time.perf_counter()
        y_pred_test, y_test, y_feature = mytest(net, test_loader)  # (42776,)
        toc2 = time.perf_counter()
        print("测试时间: {:.2f} 秒".format(toc2 - tic2))
        # Evaluation indexes 每轮的OA、AA、Kappa
        each_oa, confusion, each_acc, each_aa, each_kappa = acc_reports(y_test, y_pred_test)
        oa.append(each_oa)
        acc.append(each_acc)
        aa.append(each_aa)
        kappa.append(each_kappa)
        log.write('Test_Epoch: %.f: Each_OA: %.2f, Each_AA: %.2f, Each_kappa: %.2f \n' % (num+1, each_oa, each_aa, each_kappa))
        save_path = config.checkpoint_path + 'TestEpoch%.f_OA%.3f_AA%.3f_Kappa%.3f/' % (num+1, each_oa, each_aa, each_kappa)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # torch.save(net.state_dict(), save_path + 'TestEpoch%.f_OA%.3f_AA%.3f_Kappa%.3f.pth.tar' % (num+1, each_oa, each_aa, each_kappa))
        get_cls_map_PU.get_cls_map(net, all_data_loader, y, save_path)
        with open(save_path + "acc.txt", "w") as file:
            for item in each_acc:
                file.write("%s\n" % item)
            file.write("================================================================\n")
            file.write("%s\n" % each_oa)
            file.write("%s\n" % each_aa)
            file.write("%s\n" % each_kappa)
    # ================================================
    acc = np.array(acc)  # 转为 NumPy 数组以方便计算
    num_classes = acc.shape[1]  # 每类的准确率个数

    for class_num in range(num_classes):
        class_oa = acc[:, class_num]  # 第 class_num 类在每次实验中的准确率
        log.write('Class %d OA: %.2f, std: %.2f, var: %.2f\n' %
                  (class_num + 1, np.mean(class_oa), np.std(class_oa), np.var(class_oa)))

    log.write('   AVG:   OA: %.2f, std: %.2f, var: %.2f    AA: %.2f, std: %.2f, var: %.2f    Kappa: %.2f, std: %.2f, var: %.2f \n'
              % (np.mean(oa), np.std(oa), np.var(oa),   np.mean(aa), np.std(aa), np.var(aa),     np.mean(kappa), np.std(kappa), np.var(kappa)))
    with open((config.logs + 'AVG_OA%.3f_AA%.3f_Kappa%.3f.txt' % (np.mean(oa), np.mean(aa), np.mean(kappa))), 'w') as file:
        file.write("OA: %.2f, std: %.2f, var: %.2f\n" % (np.mean(oa), np.std(oa), np.var(oa)))
        file.write("AA: %.2f, std: %.2f, var: %.2f\n" % (np.mean(aa), np.std(aa), np.var(aa)))
        file.write("Kappa: %.2f, std: %.2f, var: %.2f\n" % (np.mean(kappa), np.std(kappa), np.var(kappa)))
        # file.write("Params: %.4f\n" % (params / 1000 ** 2))
        # file.write("Flops: %.4f\n" % (flops / 1000 ** 3))
        file.write("Training time: %.2f\n" % (toc1 - tic1))
        file.write("Testing time: %.2f\n" % (toc2 - tic2))