# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import os
import skimage.io
import tensorflow as tf

import vgg19

CONTENT_LAYER = 'conv4_2'
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

ALPHA = 1.0
BETA = 50.0
LR = 1.0

def load_image(path):
    img = skimage.io.imread(path)
    yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)  # 明度，色度，浓度
    img = img - vgg19.VGG_MEAN
    img = img[:, :, (2, 1, 0)]  # igb to bgr
    return img[np.newaxis, :, :, :], yuv

def save_image(img, path, content_yuv=None):
    img = np.squeeze(img)  # 剔除img中长度为1的轴，例：shape:[1,x,y,z]=>shape:[x,y.z]
    img = img[:, :, (2, 1, 0)]  # bgr to rgb
    img = img + vgg19.VGG_MEAN
    if content_yuv is not None:
        yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)
        yuv[:, :, 1:3] = content_yuv[:, :, 1:3]
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)  # 保留颜色
    img = np.clip(img, 0, 255).astype(np.uint8)
    skimage.io.imsave(path, img)

# 针对某层，输出该层的gram矩阵
def feature_to_gram(f):
    shape = f.get_shape().as_list()
    n_channels = shape[3]
    size = np.prod(shape)
    f = tf.reshape(f, [-1, n_channels])
    return tf.matmul(tf.transpose(f), f) / size

# 返回gram矩阵列表
def get_style_rep(vgg):
    return map(feature_to_gram, map(lambda l: getattr(vgg, l), STYLE_LAYERS))

def compute_style_loss(style_rep, image_vgg):
    style_losses = map(tf.nn.l2_loss, [a - b for (a, b) in zip(style_rep, get_style_rep(image_vgg))])
    style_losses = [style_losses[i]/style_rep[i].size for i in range(len(style_losses))]
    return reduce(tf.add, style_losses)

def main(content_path, style_path, output_dir, iterations, vgg_path, preserve_color):
    content_img, content_yuv = load_image(content_path)  # ndarray类型
    print 'content_img.shape', content_img.shape
    style_img, _ = load_image(style_path)

    with tf.Session() as sess:
        content_vgg = vgg19.Vgg19(vgg_path)
        content = tf.placeholder("float", content_img.shape)
        content_vgg.build(content)
        style_vgg = vgg19.Vgg19(vgg_path)
        style = tf.placeholder("float", style_img.shape)
        style_vgg.build(style)

        sess.run(tf.global_variables_initializer())
        # 注意：以下两rep只需一次计算
        content_rep = sess.run(getattr(content_vgg, CONTENT_LAYER), feed_dict={content: content_img})  # content的rep是该层内容
        style_rep = sess.run(get_style_rep(style_vgg), feed_dict={style: style_img})  # 注意：style的rep是gram矩阵

    # start with white noise
    noise = tf.truncated_normal(content_img.shape, stddev=0.1*np.std(content_img))
    image = tf.Variable(noise)
    image_vgg = vgg19.Vgg19(vgg_path)
    image_vgg.build(image)

    # define loss and optimizer
    content_loss = tf.nn.l2_loss(getattr(image_vgg, CONTENT_LAYER) - content_rep) / content_rep.size
    print 'content.shape:', content.shape
    style_loss = compute_style_loss(style_rep, image_vgg)
    total_loss = ALPHA * content_loss + BETA * style_loss
    optimizer = tf.train.AdamOptimizer(LR)
    optimize = optimizer.minimize(total_loss)

    # style transfer
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1, iterations+1):
            sess.run(optimize)
            fmt_str = 'Iteration {:4}/{:4}    content loss {:14}  style loss {:14}'
            print fmt_str.format(i, iterations, ALPHA*content_loss.eval(), BETA*style_loss.eval())
            output_path = os.path.join(output_dir, 'output_{:04}.jpg'.format(i))
            save_image(image.eval(), output_path, content_yuv if preserve_color else None)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--content', dest='content', default='examples/content.jpg', help='path to content image')
    # parser.add_argument('--style', dest='style', default='examples/style.jpg', help='path to style image')
    # parser.add_argument('--output', dest='output', default='output/', help='output directory')
    # parser.add_argument('--iterations', type=int, dest='iterations', default=1000, help='iterations')
    # parser.add_argument('--vgg', dest='vgg', default='vgg19.npy', help='path to pretrained vgg-19 npy model')
    # parser.add_argument('--preserve_color', dest='preserve_color', action='store_true', help='preserve color')
    # args = parser.parse_args()
    # print('Running style transfer with arguments: {}'.format(vars(args)))
    #
    # assert os.path.isfile(args.vgg), \
    #     'Pretrained vgg-19 model not found at {}. Please refer to ' \
    #     'https://github.com/machrisaa/tensorflow-vgg for download instructions.'.format(args.vgg)
    #
    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)
    # print('Saving output images to {}'.format(args.output))

    main('./content/content.jpg', './style/style.jpg', './output', 9999, './vgg19.npy', True)
