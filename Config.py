# -*- coding: utf-8 -*-


class Config:
    content_layer = 'conv4_2'
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    ALPHA = 1.0
    BETA = 50.0
    LR = 0.1
    ITERATIONS = 9999

    content_path = './content/'
    STYLE_PATH = './style/style.jpg'
    VGG_PATH = './vgg19.npy'
    OUTPUT_DIR = './output2'
    PRESERVE_COLOR = True

    batch_size = 1

    # 注意：content的batch的shape是：[batch_size, 1, height, width, channels]
    batch_shape = [None, 1, 320, 480, 3]  # 统一应用于content_input
    noise_shape = batch_shape[1:]  # 应用于noise

    img_height = batch_shape[2]
    img_width = batch_shape[3]
    num_channels = batch_shape[4]
