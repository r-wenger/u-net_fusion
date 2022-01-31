import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Concatenate, Conv2DTranspose, Dropout, Add
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from segmentation_models.losses import CategoricalCELoss

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same", use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(
        input)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same", use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)
    x = Activation("relu")(x)

    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def unet_model(im_sz, n_channels, n_classes, weights, lr, encoder_weights='imagenet'):
    """ Input """
    input_shape = (im_sz, im_sz, n_channels)
    inputs = Input(input_shape)
    dropout_value = 0.5

    """ Pre-trained VGG16 Model """
    if encoder_weights == 'imagenet':
        vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    else:
        vgg16 = VGG16(include_top=False, weights=None, input_tensor=inputs)

    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output  ## (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output  ## (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output  ## (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output  ## (64 x 64)

    """ Bridge """
    b1 = vgg16.get_layer("block5_conv3").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  ## (64 x 64)
    d1 = Dropout(dropout_value)(d1)
    d2 = decoder_block(d1, s3, 256)  ## (128 x 128)
    d2 = Dropout(dropout_value)(d2)
    d3 = decoder_block(d2, s2, 128)  ## (256 x 256)
    d3 = Dropout(dropout_value)(d3)
    d4 = decoder_block(d3, s1, 64)  ## (512 x 512)
    d4 = Dropout(dropout_value)(d4)

    """ Output """
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(d4)

    loss = CategoricalCELoss(class_weights=weights)

    model = Model(inputs, outputs, name="VGG16_U-Net")

    def dice(y_true, y_pred, smooth=1.):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    model.compile(optimizer=Adam(lr),
                  loss=loss,
                  metrics=['categorical_accuracy', dice])
    return model


def unet_model_decoder_fusion(im_sz, n_channels, n_classes, weights, lr):
    """ Input """
    input_shape = (im_sz, im_sz, n_channels)
    inputs_1 = Input(input_shape)
    inputs_2 = Input(input_shape)
    dropout_value = 0.5

    """ Metric """

    def dice(y_true, y_pred, smooth=1.):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    """ Loss """
    loss = CategoricalCELoss(class_weights=weights)

    """ Define Model """
    vgg16_1 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs_1)
    for layer in vgg16_1.layers:
        layer._name = layer.name + str('_1')
    vgg16_2 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs_2)
    for layer in vgg16_2.layers:
        layer._name = layer.name + str('_2')

    """ NETWORK 1 """
    """ Extracting features model 1 """
    f1 = vgg16_1.get_layer("block1_conv2_1").output
    f2 = vgg16_1.get_layer("block2_conv2_1").output
    f3 = vgg16_1.get_layer("block3_conv3_1").output
    f4 = vgg16_1.get_layer("block4_conv3_1").output

    """ Bridge 1 """
    f5 = vgg16_1.get_layer("block5_conv3_1").output

    """ Decoder """
    de1 = decoder_block(f5, f4, 512)  ## (64 x 64)
    de1 = Dropout(dropout_value)(de1)
    de2 = decoder_block(de1, f3, 256)  ## (128 x 128)
    de2 = Dropout(dropout_value)(de2)
    de3 = decoder_block(de2, f2, 128)  ## (256 x 256)
    de3 = Dropout(dropout_value)(de3)
    de4 = decoder_block(de3, f1, 64)  ## (512 x 512)
    de4 = Dropout(dropout_value)(de4)

    """ NETWORK 2 """
    """ Encoder """
    s1 = vgg16_2.get_layer("block1_conv2_2").output  ## (512 x 512)
    s2 = vgg16_2.get_layer("block2_conv2_2").output  ## (256 x 256)
    s3 = vgg16_2.get_layer("block3_conv3_2").output  ## (128 x 128)
    s4 = vgg16_2.get_layer("block4_conv3_2").output  ## (64 x 64)

    """ Bridge 2 """
    b1 = vgg16_2.get_layer("block5_conv3_2").output  ## (32 x 32)
    b1 = Add()([b1, f5])

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  ## (64 x 64)
    d1 = Dropout(dropout_value)(d1)
    d1 = Add()([d1, de1])
    d2 = decoder_block(d1, s3, 256)  ## (128 x 128)
    d2 = Dropout(dropout_value)(d2)
    d2 = Add()([d2, de2])
    d3 = decoder_block(d2, s2, 128)  ## (256 x 256)
    d3 = Dropout(dropout_value)(d3)
    d3 = Add()([d3, de3])
    d4 = decoder_block(d3, s1, 64)  ## (512 x 512)
    d4 = Dropout(dropout_value)(d4)
    d4 = Add()([d4, de4])

    """ Output """
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(d4)

    model = Model([inputs_1, inputs_2], outputs, name="VGG16_U-Net_encoder_fusion")

    model.compile(optimizer=Adam(lr),
                  loss=loss,
                  metrics=['categorical_accuracy', dice])
    return model

def unet_model_encoder_fusion(im_sz, n_channels, n_classes, weights, lr):
    """ Input """
    input_shape = (im_sz, im_sz, n_channels)
    inputs_1 = Input(input_shape)
    inputs_2 = Input(input_shape)
    dropout_value = 0.5

    """ Metric """

    def dice(y_true, y_pred, smooth=1.):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    """ Loss """
    loss = CategoricalCELoss(class_weights=weights)

    """ Define Model """
    vgg16_1 = VGG16(include_top=False, weights=None, input_tensor=inputs_1)
    for layer in vgg16_1.layers:
        layer._name = layer.name + str('_1')
    vgg16_2 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs_2)
    for layer in vgg16_2.layers:
        layer._name = layer.name + str('_2')

    """ Extracting features model 1 """
    f1 = vgg16_1.get_layer("block1_conv2_1").output
    f2 = vgg16_1.get_layer("block2_conv2_1").output
    f3 = vgg16_1.get_layer("block3_conv3_1").output
    f4 = vgg16_1.get_layer("block4_conv3_1").output
    f5 = vgg16_1.get_layer("block5_conv3_1").output

    """ Encoder """
    s1 = vgg16_2.get_layer("block1_conv2_2").output  ## (512 x 512)
    s1 = Add()([s1, f1])
    s2 = vgg16_2.get_layer("block2_conv2_2").output  ## (256 x 256)
    s2 = Add()([s2, f2])
    s3 = vgg16_2.get_layer("block3_conv3_2").output  ## (128 x 128)
    s3 = Add()([s3, f3])
    s4 = vgg16_2.get_layer("block4_conv3_2").output  ## (64 x 64)
    s4 = Add()([s4, f4])

    """ Bridge """
    b1 = vgg16_2.get_layer("block5_conv3_2").output  ## (32 x 32)
    b1 = Add()([b1, f5])

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  ## (64 x 64)
    d1 = Dropout(dropout_value)(d1)
    d2 = decoder_block(d1, s3, 256)  ## (128 x 128)
    d2 = Dropout(dropout_value)(d2)
    d3 = decoder_block(d2, s2, 128)  ## (256 x 256)
    d3 = Dropout(dropout_value)(d3)
    d4 = decoder_block(d3, s1, 64)  ## (512 x 512)
    d4 = Dropout(dropout_value)(d4)

    """ Output """
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(d4)

    model = Model([inputs_1, inputs_2], outputs, name="VGG16_U-Net_encoder_fusion")

    model.compile(optimizer=Adam(lr),
                  loss=loss,
                  metrics=['categorical_accuracy', dice])
    return model
