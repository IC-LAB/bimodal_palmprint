# coding:utf-8

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam,SGD,RMSprop
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.core import Activation, Dropout
from keras.layers.merge import concatenate
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D


def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    return Conv2D(f,(k,k),padding=border_mode,strides=(s,s))


def Deconvolution(f, output_shape, k=2, s=2, **kwargs):
    """Convenience method for Transposed Convolutions."""
    return Conv2DTranspose(f,(k,k),strides=(s,s),padding='same')


def BatchNorm(mode=2, axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    return BatchNormalization(axis=-1)

def generator(nf, batchSize, name='unet'):
    i = Input(shape=(256, 256, 3))
    # 3 x 256 x 256

    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm(mode=0)(conv1)
    x = LeakyReLU(0.2)(conv1)
    # nf x 128 x 128

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm(mode=0)(conv2)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 64x 64

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm(mode=0)(conv3)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 32 x 32

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm(mode=0)(conv4)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 16 x 16

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm(mode=0)(conv5)
    x = LeakyReLU(0.2)(conv5)
    # nf*8 x 8 x 8

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm(mode=0)(conv6)
    x = LeakyReLU(0.2)(conv6)
    # nf*8 x 4 x 4

    conv7 = Convolution(nf * 8)(x)
    conv7 = BatchNorm(mode=0)(conv7)
    x = LeakyReLU(0.2)(conv7)
    # nf*8 x 2 x 2

    conv8 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv8 = BatchNorm(mode=0)(conv8)
    x = LeakyReLU(0.2)(conv8)
    # nf*8 x 1 x 1

    deconv1 = Deconvolution(nf * 8, (None, 2, 2, nf * 8), k=2, s=2)(x)
    deconv1 = BatchNorm(mode=0)(deconv1)
    deconv1 = Dropout(0.5)(deconv1)
    x = concatenate([deconv1,conv7],axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*8 + nf*8) x 2 x 2

    deconv2 = Deconvolution(nf * 8, (None, 4, 4, nf * 8))(x)
    deconv2 = BatchNorm(mode=0)(deconv2)
    deconv2 = Dropout(0.5)(deconv2)
    x = concatenate([deconv2, conv6], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*8 + nf*8) x 4 x 4

    deconv3 = Deconvolution(nf * 8, (None, 8, 8, nf * 8))(x)
    deconv3 = BatchNorm(mode=0)(deconv3)
    deconv3 = Dropout(0.5)(deconv3)
    x = concatenate([deconv3, conv5], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*8 + nf*8) x 8 x 8

    deconv4 = Deconvolution(nf * 8, (None, 16, 16, nf * 8))(x)
    deconv4 = BatchNorm(mode=0)(deconv4)
    x = concatenate([deconv4, conv4], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*8 + nf*8) x 16 x 16

    deconv5 = Deconvolution(nf * 4, (None, 32, 32, nf * 4))(x)
    deconv5 = BatchNorm(mode=0)(deconv5)
    x = concatenate([deconv5, conv3], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*4 + nf*4) x 32 x 32

    deconv6 = Deconvolution(nf * 2, (None, 64, 64, nf * 2))(x)
    deconv6 = BatchNorm(mode=0)(deconv6)
    x = concatenate([deconv6, conv2], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*2 + nf*2) x 64 x 64

    deconv7 = Deconvolution(nf * 2, (None, 128, 128, nf * 2))(x)
    deconv7 = BatchNorm(mode=0)(deconv7)
    x = concatenate([deconv7, conv1], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*2 + nf*2) x 128 x 128

    deconv8 = Deconvolution(3, (None, 256, 256, 3))(x)
    # 3 x 256 x 256

    out = Activation('tanh')(deconv8)

    unet = Model(i, out, name=name)

    return unet


def discriminator(nf, opt=RMSprop(lr=0.00005), name='d'):
    i = Input(shape=(256, 256, 3 + 3))
    # (3 + 3) x 256 x 256

    conv1 = Convolution(nf * 1)(i)
    x = LeakyReLU(0.2)(conv1)
    # nf*1 x 128x 128

    conv2 = Convolution(nf * 2)(x)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 64 x 64

    conv3 = Convolution(nf * 4)(x)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 32 x 32

    conv4 = Convolution(nf * 8)(x)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 16 x 16

    conv5 = Convolution(1)(x)
    # 1 x 8 x 8

    out = GlobalAveragePooling2D()(conv5)

    d = Model(i, out, name=name)

    def d_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    d.compile(optimizer=opt, loss=d_loss)
    return d


def pix2pix(atob, d, alpha=0,belta=0.1, opt=RMSprop(lr=0.00005), name='pix2pix'):
    a = Input(shape=(256, 256, 3))
    b = Input(shape=(256, 256, 3))

    # A -> B'
    bp = atob(a)

    # Discriminator receives the pair of images
    d_in = concatenate([a,bp],axis=-1)

    pix2pix = Model([a, b], d(d_in), name=name)

    def pix2pix_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # Adversarial Loss
        # L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)
        L_adv = K.mean(y_true * y_pred)

        # A to B loss
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        L_atob = K.mean(K.abs(b_flat - bp_flat))

        return alpha*L_adv + belta * L_atob

    # This network is used to train the generator. Freeze the discriminator part.
    pix2pix.get_layer('d').trainable = False

    pix2pix.compile(optimizer=opt, loss=pix2pix_loss)
    return pix2pix