# -*- coding: utf-8 -*-


class Config:
    content_layer = 'conv4_2'
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    vgg_path = './vgg19.npy'


class StyleTransferConfig(Config):
    ALPHA = 1.0
    BETA = 50.0
    LR = 1.0
    ITERATIONS = 9999
    content_path = './content_neural/content.jpg'
    style_path = './style_neural/style.jpg'
    output = './output_neural'
    preserve_color = True


class FastStyleTransferConfig(Config):
    ALPHA = 1.0
    BETA = 50.0
    LR = 0.1
    ITERATIONS = 9999
    model_path = './model/baby_model'
    content_train_path = './content_fast_train/'
    content_transfer_path = './content_fast_transfer/content.jpg'
    style_path = './style_fast/style.jpg'
    output = './output_fast'
    preserve_color = True
    batch_size = 1

    # 注意：content的batch的shape是：[batch_size, 1, height, width, channels]
    batch_shape = [None, 1, 320, 480, 3]  # 统一应用于content_input
    noise_shape = batch_shape[1:]  # 应用于noise

    img_height = batch_shape[2]
    img_width = batch_shape[3]
    num_channels = batch_shape[4]



