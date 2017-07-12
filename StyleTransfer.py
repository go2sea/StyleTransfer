# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import os
import skimage.io
import tensorflow as tf
import functools
from Config import Config

import vgg19

def lazy_property(func):
    attribute = '_lazy_' + func.__name__
    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper


class StyleTransfer:
    def __init__(self, config):
        self.sess = tf.Session()

        self.config = config

        self.content_img_bgr, self.content_img_yuv = self.load_image(self.config.CONTENT_PATH)
        self.style_img_bgr, _ = self.load_image(self.config.STYLE_PATH)

        self.content_input = tf.placeholder(tf.float32, self.content_img_bgr.shape)
        self.style_input = tf.placeholder(tf.float32, self.style_img_bgr.shape)

        self.content_vgg = vgg19.Vgg19(self.config.VGG_PATH)
        self.content_vgg.build(self.content_input)
        self.style_vgg = vgg19.Vgg19(self.config.VGG_PATH)
        self.style_vgg.build(self.style_input)

        self.sess.run(tf.global_variables_initializer())

        # 注意：以下两rep只需被计算一次
        self.content_rep = self.sess.run(getattr(self.content_vgg, self.config.CONTENT_LAYER),
                                         feed_dict={self.content_input: self.content_img_bgr})
        self.style_rep = self.sess.run(self.get_style_rep(self.style_vgg),
                                       feed_dict={self.style_input: self.style_img_bgr})

        # 从白噪声开始（noise是变量）
        self.noise = tf.Variable(tf.truncated_normal(self.content_img_bgr.shape, stddev=0.1*np.std(self.content_img_bgr)))
        self.noise_vgg = vgg19.Vgg19(self.config.VGG_PATH)
        self.noise_vgg.build(self.noise)

        self.content_loss
        self.style_loss
        self.optimize

        self.sess.run(tf.global_variables_initializer())

    def transfer_train(self):
        for i in range(0, self.config.ITERATIONS):
            self.sess.run(self.optimize)
            fmt_str = 'Iteration {:4}/{:4}    content loss {:14}  style loss {:14}'
            print fmt_str.format(i,
                                 self.config.ITERATIONS,
                                 self.config.ALPHA*self.sess.run(self.content_loss),
                                 self.config.BETA*self.sess.run(self.style_loss))
            output_path = os.path.join(self.config.OUTPUT_DIR, 'output_{:04}.jpg'.format(i))
            self.save_image(self.sess.run(self.noise), output_path, self.content_img_yuv if self.config.PRESERVE_COLOR else None)

    @lazy_property
    def content_loss(self):
        return tf.nn.l2_loss(getattr(self.noise_vgg, self.config.CONTENT_LAYER) - self.content_rep) / self.content_rep.size

    @lazy_property
    def style_loss(self):
        style_losses = map(tf.nn.l2_loss, [a - b for (a, b) in zip(self.style_rep, self.get_style_rep(self.noise_vgg))])
        style_losses = [style_losses[i]/self.style_rep[i].size for i in range(len(style_losses))]
        return reduce(tf.add, style_losses)

    @lazy_property
    def optimize(self):
        total_loss = self.config.ALPHA * self.content_loss + self.config.BETA * self.style_loss
        optimizer = tf.train.AdamOptimizer(self.config.LR)
        return optimizer.minimize(total_loss)

    def get_style_rep(self, vgg):
        return map(self.feature_to_gram, map(lambda l: getattr(vgg, l), self.config.STYLE_LAYERS))

    def feature_to_gram(self, f):
        shape = f.get_shape().as_list()
        print 'shape:', shape
        num_channels = shape[3]
        size = np.prod(shape[:-1])  # 注意：size应为feature map中元素个数
        # size = np.prod(shape)  # 注意：size应为feature map中元素个数
        f = tf.reshape(f, [-1, num_channels])
        return tf.matmul(tf.transpose(f), f) / size  # grapm矩阵：[num_channels, num_channels]

    def load_image(self, path):
        img = skimage.io.imread(path)
        yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)  # 明度，色度，浓度
        img = img - vgg19.VGG_MEAN
        img = img[:, :, (2, 1, 0)]  # igb to bgr
        return img[np.newaxis, :, :, :], yuv

    def save_image(self, img, path, content_yuv=None):
        img = np.squeeze(img)  # 剔除img中长度为1的轴，例：shape:[1,x,y,z]=>shape:[x,y.z]
        img = img[:, :, (2, 1, 0)]  # bgr to rgb
        img = img + vgg19.VGG_MEAN
        if content_yuv is not None:
            yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)
            yuv[:, :, 1:3] = content_yuv[:, :, 1:3]
            img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)  # 保留颜色
        img = np.clip(img, 0, 255).astype(np.uint8)
        skimage.io.imsave(path, img)


if __name__ == "__main__":
    transfer = StyleTransfer(Config())
    transfer.transfer_train()


