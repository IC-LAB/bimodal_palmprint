# coding:utf-8

import os
import numpy as np
import model as m
from tqdm import tqdm
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import copy
from PIL import Image
import pickle
import random
from keras.preprocessing.image import ImageDataGenerator

iDir = '../result/NormEnhanceLarge'
lossDir = iDir + '/loss'
weightDir = iDir + '/weights'
generateDir = iDir + '/value'
if (not os.path.exists(iDir)):
    os.makedirs(iDir)
if (not os.path.exists(lossDir)):
    os.makedirs(lossDir)
if (not os.path.exists(weightDir)):
    os.makedirs(weightDir)
if (not os.path.exists(generateDir)):
    os.makedirs(generateDir)

c = 0.01
epochs = 20
batchSize = 10
batchNum = 100
lr = 0.00005
natob = 32
nfd = 16
startNum = 0

dopt = RMSprop(lr)
# Generator
generator = m.generator(natob, batchSize=64)
# Discriminator
discriminator = m.discriminator(nfd, opt=dopt)
doutSize = discriminator.output_shape[1:-1]
pix2pix = m.pix2pix(generator, discriminator, alpha=0,belta=1, opt=dopt)

# loss
listDisLossReal = []
listDisLossGen = []
listGenLoss = []

datagen = datagen = ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='constant')
# load train data
pkl_file = open('../data/NormEnhance/train/trainX.pkl', 'rb')
trainX = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('../data/NormEnhance/train/trainY.pkl', 'rb')
trainY = pickle.load(pkl_file)
pkl_file.close()
trainData = np.concatenate((trainX,trainY),axis=-1)
gen = datagen.flow(trainData,batch_size=10)

len = np.size(trainX,0)
idx = np.arange(0,len)
# load value data
pkl_file = open('../data/NormEnhance/value/valueX.pkl', 'rb')
valX = pickle.load(pkl_file)
pkl_file.close()



pretrained = 1
trained_epochs = 80
# load pretrained weights if exists
if(pretrained==1):
    generator.load_weights(weightDir + '/generator_' + str(trained_epochs))
    discriminator.load_weights(weightDir+ '/discriminator_' + str(trained_epochs))
    pix2pix.load_weights(weightDir + '/pix2pix_' + str(trained_epochs))

# start train
for newEpoch in range(epochs):
    epoch = newEpoch + trained_epochs
    if (epoch < 10):
        ite = 20
    else:
        ite = 5
    print('epoch:' + str(epoch) + '/' + str(epochs+trained_epochs))
    for index in tqdm(range(batchNum)):
        for i in range(ite):
            # Clip D weights
            for l in discriminator.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -c, c) for w in weights]
                l.set_weights(weights)
            #######################################
            # train D model
            #######################################
            # real samples
            batch = next(gen)
            realBatchA = batch[:,:,:,0:3]
            realBatchB = batch[:,:,:,3:6]
            realBatch = np.concatenate((realBatchA, realBatchB), axis=-1)
            # generate fake samples
            fakeBatchB = generator.predict(realBatchA)
            fakeBatch = np.concatenate((realBatchA, fakeBatchB), axis=-1)
            # update D model
            discriminator.trainable = True
            listDisLossReal.append(discriminator.train_on_batch(realBatch, -np.ones(realBatch.shape[0])))
            listDisLossGen.append(discriminator.train_on_batch(fakeBatch, np.ones(fakeBatch.shape[0])))

        #######################################
        # train G model
        #######################################
        # real samples
        batch = next(gen)
        realBatchA = batch[:, :, :, 0:3]
        realBatchB = batch[:, :, :, 3:6]
        # update G model
        discriminator.trainable = False
        listGenLoss.append(pix2pix.train_on_batch([realBatchA, realBatchB], -np.ones(realBatchA.shape[0])))

    # save
    if ((epoch + 1) % 1 == 0):
        # loss 
        disLossReal = copy.deepcopy(np.array(listDisLossReal))
        disLossGen = copy.deepcopy(np.array(listDisLossGen))
        genLoss = copy.deepcopy(np.array(listGenLoss))
    
        output = open(lossDir + "/Loss_" + str(epoch + 1) + '.pkl', 'wb')
        pickle.dump(disLossReal, output)
        pickle.dump(disLossGen, output)
        pickle.dump(genLoss, output)
        output.close()
        # DLoss
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(0, disLossReal.shape[0]), disLossReal, label="disLossReal", color="red")
        plt.plot(np.arange(0, disLossGen.shape[0]), disLossGen, label="disLossGen", color="green")
        plt.legend()
        plt.savefig(lossDir + "/disLoss_" + str(epoch + 1) + ".jpg")
        # GLoss
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(0, genLoss.shape[0]), genLoss, label="genLoss", color="blue")
        plt.legend()
        plt.savefig(lossDir + "/genLoss_" + str(epoch + 1) + ".jpg")

        # value
        valY = generator.predict(valX)
        for valIndex in range(valY.shape[0]):
            img = valY[valIndex, :, :, 0]
            img = (img * 127.5) + 127.5
            img = Image.fromarray(np.uint8(img))

            if (valIndex + 1 < 10):
                newFileName = str(epoch + 1 + startNum) + "_000" + str(valIndex + 1) + ".bmp"
            elif (valIndex + 1 < 100):
                newFileName = str(epoch + 1 + startNum) + "_00" + str(valIndex + 1) + ".bmp"
            elif (valIndex + 1 < 1000):
                newFileName = str(epoch + 1 + startNum) + "_0" + str(valIndex + 1) + ".bmp"
            else:
                newFileName = str(epoch + 1 + startNum) + "_" + str(valIndex + 1) + ".bmp"

            img.save(generateDir + '/' + newFileName)

    # save checkpoints
    if ((epoch + 1) % 10 == 0):
        # save Weights
        generator.save_weights(weightDir + '/generator_' + str(epoch + 1 + startNum), overwrite=True)
        discriminator.save_weights(weightDir + '/discriminator_' + str(epoch + 1 + startNum), overwrite=True)
        pix2pix.save_weights(weightDir + '/pix2pix_' + str(epoch + 1 + startNum), overwrite=True)
