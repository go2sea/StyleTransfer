# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import os
import skimage.io
import tensorflow as tf
import vgg19
import glob
from Config import Config
from myUtils import lazy_property, conv, residual, instance_norm, resize_conv


class FastStyleTransfer:
    def __init__(self, config):
        self.sess = tf.Session()
        self.config = config
        # 由于风格固定，__init__可以加载风格图
        self.style_img_bgr, _ = self.load_image(self.config.STYLE_PATH)
        # content_vggs，注意shape中添加batch_size
        self.content_input = tf.placeholder(tf.float32, self.config.batch_shape)
        self.content_vggs = []
        for i in range(self.config.batch_size):
            self.content_vggs.append(vgg19.Vgg19(self.config.VGG_PATH))
            self.content_vggs[i].build(self.content_input[i])
        # style_vgg
        self.style_input = tf.placeholder(tf.float32, self.style_img_bgr.shape)  # 注意：style的shape不遵循config
        self.style_vgg = vgg19.Vgg19(self.config.VGG_PATH)
        self.style_vgg.build(self.style_input)
        # noise_vgg
        self.noise_input = tf.placeholder(tf.float32, [1, None, None, self.config.num_channels])
        self.noise = self.sess.run(tf.truncated_normal(self.config.noise_shape, stddev=0.001))
        self.trans_net_output_vgg = vgg19.Vgg19(self.config.VGG_PATH)
        self.trans_net_output_vgg.build(self.trans_net_output)

        self.sess.run(tf.global_variables_initializer())

        # 内容表示 & 风格表示
        self.content_reps = []
        for i in range(self.config.batch_size):
            self.content_reps.append(getattr(self.content_vggs[i], self.config.content_layer))
        # self.content_rep = getattr(self.content_vgg, self.config.CONTENT_LAYER)
        self.style_rep = self.sess.run(self.get_style_rep(self.style_vgg),
                                       feed_dict={self.style_input: self.style_img_bgr})   # 注意：风格图的风格表示只需被计算一次
        self.noise_content_rep = getattr(self.trans_net_output_vgg, self.config.content_layer)
        self.noise_style_rep = self.get_style_rep(self.trans_net_output_vgg)

        self.content_loss
        self.style_loss
        self.optimize

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_layers(self, input_imgs, w_i, b_i, training=True):
        with tf.variable_scope('conv1'):
            relu1 = tf.nn.relu(instance_norm(conv(input_imgs, [9, 9, 3, 32], None, [1, 1, 1, 1], w_i, b_i)))
        with tf.variable_scope('conv2'):
            relu2 = tf.nn.relu(instance_norm(conv(relu1, [3, 3, 32, 64], None, [1, 2, 2, 1], w_i, b_i)))
        with tf.variable_scope('conv3'):
            relu3 = tf.nn.relu(instance_norm(conv(relu2, [3, 3, 64, 128], None, [1, 2, 2, 1], w_i, b_i)))
        with tf.variable_scope('residual1'):
            residual1 = residual(relu3, [3, 3, 128, 128], None, [1, 1, 1, 1], w_i, b_i)
        with tf.variable_scope('residual2'):
            residual2 = residual(residual1, [3, 3, 128, 128], None, [1, 1, 1, 1], w_i, b_i)
        with tf.variable_scope('residual3'):
            residual3 = residual(residual2, [3, 3, 128, 128], None, [1, 1, 1, 1], w_i, b_i)
        with tf.variable_scope('residual4'):
            residual4 = residual(residual3, [3, 3, 128, 128], None, [1, 1, 1, 1], w_i, b_i)
        with tf.variable_scope('deconv1'):
            deconv1 = tf.nn.relu(instance_norm(resize_conv(residual4, [3, 3, 128, 64], None, [1, 2, 2, 1], w_i, b_i, None, training)))
        with tf.variable_scope('deconv2'):
            deconv2 = tf.nn.relu(instance_norm(resize_conv(deconv1, [3, 3, 64, 32], None, [1, 2, 2, 1], w_i, b_i, None, training)))
        with tf.variable_scope('deconv3'):
            deconv3 = tf.nn.relu(instance_norm(conv(deconv2, [9, 9, 32, 3], None, [1, 1, 1, 1], w_i, b_i, None)))
        return deconv3

    @lazy_property
    def trans_net_output(self):
        w_initializer = tf.random_uniform_initializer(-0.1, 0.1)
        b_initializer = tf.constant_initializer(0.1)
        return self.build_layers(self.noise, w_initializer, b_initializer)

    # 使用训练好的model进行风格化
    def transfer(self):
        self.restore_sess()
        noise_bgr, noise_yuv = self.load_image(self.config.content_transfer_path)

        output_path = os.path.join(self.config.fast_style_transfer_output, 'transfered.jpg')
        noise_img_bgr = self.sess.run(tf.image.resize_images(noise_bgr,
                                                             size=[self.config.img_height, self.config.img_width],
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        noise_img_yuv = self.sess.run(tf.image.resize_images(noise_bgr,
                                                             size=[self.config.img_height, self.config.img_width],
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))

        print 'noise_img_bgr.shape:', noise_img_bgr.shape
        print 'noise_img_yuv.shape:', noise_img_yuv.shape
        self.save_image(self.sess.run(self.trans_net_output, feed_dict={
            self.noise_input: noise_img_bgr
        }), output_path, noise_img_yuv if self.config.preserve_color else None)

    def train_transfer_net(self):
        self.save_sess()
        self.restore_sess()
        content_img_paths = sorted(glob.glob("{}/*".format(self.config.content_train_path)))
        index_list = np.random.randint(0, len(content_img_paths), size=self.config.batch_size)

        batch = []
        for i in range(self.config.batch_size):
            content_img_path = content_img_paths[index_list[i]]
            content_img_bgr, content_img_yuv = self.load_image(content_img_path)
            # reshape至config.shape，注意resize_images的input需为4-D or 3-D的tensor
            resized = self.sess.run(tf.image.resize_images(content_img_bgr,
                                                           size=[self.config.img_height, self.config.img_width],
                                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
            batch.append(resized)

        for i in range(0, self.config.ITERATIONS):
            fmt_str = 'Iteration {:4}/{:4}    content loss {:14}  style loss {:14}'
            # print fmt_str.format(i, self.config.ITERATIONS,
            #                      self.config.ALPHA*self.sess.run(self.content_loss, feed_dict={
            #                          self.content_input: batch
            #                      }),
            #                      self.config.BETA*self.sess.run(self.style_loss, feed_dict={
            #                          self.content_input: batch
            #                      }))
            self.sess.run(self.optimize, feed_dict={
                self.content_input: batch,
                self.noise_input: self.noise
            })

            fmt_str = 'Iteration {:4}/{:4}    content loss {:14}  style loss {:14}'
            print fmt_str.format(i, self.config.ITERATIONS,
                                 self.config.ALPHA*self.sess.run(self.content_loss, feed_dict={
                                     self.content_input: batch,
                                     self.noise_input: self.noise
                                 }),
                                 self.config.BETA*self.sess.run(self.style_loss, feed_dict={
                                     self.content_input: batch,
                                     self.noise_input: self.noise
                                 }))
            # output_path = os.path.join(self.config.OUTPUT_DIR, 'output_{:04}.jpg'.format(i))
            # self.save_image(self.sess.run(self.noise), output_path, content_img_yuv if self.config.PRESERVE_COLOR else None)

    @lazy_property
    def content_loss(self):
        # loss = tf.nn.l2_loss(self.noise_content_rep - self.content_rep)
        losses = [tf.nn.l2_loss((self.noise_content_rep - cr)) for cr in self.content_reps]
        # return losses[0]
        # print 'sahpe:', tf.shape(losses)
        losses = tf.div(losses, tf.to_float(tf.reduce_prod(tf.shape(self.content_reps[0])[:-1])))
        return tf.reduce_sum(losses) / self.config.batch_size

    @lazy_property
    def style_loss(self):
        style_losses = map(tf.nn.l2_loss, [a - b for (a, b) in zip(self.style_rep, self.noise_style_rep)])
        style_losses = [style_losses[i]/self.style_rep[i].size for i in range(len(style_losses))]
        return reduce(tf.add, style_losses)

    @lazy_property
    def optimize(self):
        total_loss = self.config.ALPHA * self.content_loss + self.config.BETA * self.style_loss
        optimizer = tf.train.AdamOptimizer(self.config.LR)
        return optimizer.minimize(total_loss)

    def get_style_rep(self, vgg):
        return map(self.feature_to_gram, map(lambda l: getattr(vgg, l), self.config.style_layers))

    def feature_to_gram(self, f):
        # shape = tf.get_shape(f)
        shape = tf.shape(f)  # 一些运行时shape才能确定的tensor，要用tf.shape
        num_channels = shape[3]
        # size = np.prod(shape[:-1])  # 注意：size应为feature map中元素个数
        size = tf.reduce_prod(shape[:-1])  # 注意：size应为feature map中元素个数，对于tensor应使用tf.reduce_prod
        f = tf.reshape(f, [-1, num_channels])
        return tf.matmul(tf.transpose(f), f) / tf.to_float(size)  # grapm矩阵：[num_channels, num_channels]

    def load_image(self, path):
        img = skimage.io.imread(path)
        yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)  # 明度，色度，浓度
        img = img - vgg19.VGG_MEAN
        img = img[:, :, (2, 1, 0)]  # igb to bgr
        return img[np.newaxis, :, :, :], yuv

    def save_image(self, img, path, content_yuv=None):
        print 'img.shape:', img.shape
        img = np.squeeze(img)  # 剔除img中长度为1的轴，例：shape:[1,x,y,z]=>shape:[x,y.z]
        img = img[:, :, (2, 1, 0)]  # bgr to rgb
        img = img + vgg19.VGG_MEAN
        if content_yuv is not None:
            yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)
            print 'yuv.shape:', yuv.shape
            print 'content_yuv:', content_yuv.shape
            yuv[:, :, 1:3] = content_yuv[0, :, :, 1:3]
            img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)  # 保留颜色
        img = np.clip(img, 0, 255).astype(np.uint8)
        skimage.io.imsave(path, img)

    def save_sess(self):
        self.saver.save(self.sess, self.config.model_path)
        print "Sess saved in ", self.config.model_path, ' !!!'

    def restore_sess(self):
        self.saver.restore(self.sess, self.config.model_path)
        print 'Sess restored from ', self.config.model_path, ' !!!'

if __name__ == "__main__":
    transfer = FastStyleTransfer(Config())
    # transfer.train_transfer_net()
    transfer.transfer()


