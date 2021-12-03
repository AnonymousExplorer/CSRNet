import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Input,ZeroPadding2D,Add,Dense,Activation,BatchNormalization,MaxPool2D,UpSampling2D, Concatenate
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop,SGD
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K
import os
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint


def create_hourglass_network(num_stacks, num_channels, bottleneck,channel,weight):
    #input = Input(shape=(inres[0], inres[1], inres[2]))

    #front_features = create_front_module(input, num_channels, bottleneck)
    input_img = Input([None,None,3])
    head_next_stage = input_img

    #outputs = []
    for i in range(num_stacks):
        #head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, num_channels, bottleneck, i)
        head_next_stage = hourglass_module(head_next_stage, num_channels, bottleneck, i)
        #outputs.append(head_to_loss)
    output = Conv2D(channel, kernel_size=(1, 1), padding='same')(head_next_stage)

    #model = Model(inputs=input, outputs=output)
    #rms = RMSprop(lr=5e-4)
    #model.compile(optimizer=rms, loss=mean_squared_error, metrics=["accuracy"])
    model = Model(input_img,output)
    if weight:
        model.load_weights(weight)
    return model
def bottleneck_block(bottom, num_out_channels, block_name='a'):
    # skip layer
    if K.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same')(bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = Conv2D(num_out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same')(bottom)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same')(_x)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same')(_x)
    _x = BatchNormalization()(_x)
    _x = Add()([_skip, _x])
    return _x

def hourglass_module(bottom, num_channels, bottleneck, hgid):
    # create left features , f1, f2, f4, and f8
    left_features = create_left_half_blocks(bottom, bottleneck, hgid, num_channels)

    # create right features, connect with left features
    rf1 = create_right_half_blocks(left_features, bottleneck, hgid, num_channels)

    # add 1x1 conv with two heads, head_next_stage is sent to next stage
    # head_parts is used for intermediate supervision
    #head_next_stage, head_parts = create_heads(bottom, rf1, num_classes, hgid, num_channels)
    head_next_stage = create_heads(bottom, rf1, hgid, num_channels)

    return head_next_stage#, head_parts



def create_front_module(input, num_channels, bottleneck):
    # front module, input to 1/4 resolution
    # 1 7x7 conv + maxpooling
    # 3 residual block

    _x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='front_conv_1x1_x1')(
        input)
    _x = BatchNormalization()(_x)

    _x = bottleneck(_x, num_channels // 2)
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)

    _x = bottleneck(_x, num_channels // 2)
    _x = bottleneck(_x, num_channels)

    return _x


def create_left_half_blocks(bottom, bottleneck, hglayer, num_channels):
    # create left half blocks for hourglass module
    # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

    hgname = 'hg' + str(hglayer)
    # print(bottom)
    f1 = bottleneck(bottom, num_channels)
    # print(bottleneck)
    # print(f1)
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

    f2 = bottleneck(_x, num_channels)
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

    f4 = bottleneck(_x, num_channels)
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

    f8 = bottleneck(_x, num_channels)

    return (f1, f2, f4, f8)


def connect_left_to_right(left, right, bottleneck, num_channels):
    '''
    :param left: connect left feature to right feature
    :param name: layer name
    :return:
    '''
    # left -> 1 bottlenect
    # right -> upsampling
    # Add   -> left + right

    _xleft = bottleneck(left, num_channels)
    _xright = UpSampling2D()(right)
    add = Add()([_xleft, _xright])
    out = bottleneck(add, num_channels)
    return out


def bottom_layer(lf8, bottleneck, hgid, num_channels):
    # blocks in lowest resolution
    # 3 bottlenect blocks + Add

    lf8_connect = bottleneck(lf8, num_channels)

    _x = bottleneck(lf8, num_channels)
    _x = bottleneck(_x, num_channels)
    _x = bottleneck(_x, num_channels)

    rf8 = Add()([_x, lf8_connect])
    rf8 = bottleneck(rf8, num_channels)
    return rf8


def create_right_half_blocks(leftfeatures, bottleneck, hglayer, num_channels):
    lf1, lf2, lf4, lf8 = leftfeatures

    rf8 = bottom_layer(lf8, bottleneck, hglayer, num_channels)

    rf4 = connect_left_to_right(lf4, rf8, bottleneck, num_channels)

    rf2 = connect_left_to_right(lf2, rf4, bottleneck, num_channels)

    rf1 = connect_left_to_right(lf1, rf2, bottleneck, num_channels)

    return rf1


def create_heads(prelayerfeatures, rf1, hgid, num_channels):
    # two head, one head to next stage, one head to intermediate features
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='relu', padding='same')(rf1)
    head = BatchNormalization()(head)

    # for head as intermediate supervision, use 'linear' as activation.
    #head_parts = Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
    #                    name=str(hgid) + '_conv_1x1_parts')(head)

    # use linear activation
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same')(head)
    #head_m = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
    #                name=str(hgid) + '_conv_1x1_x3')(head_parts)

    #head_next_stage = Add()([head, head_m, prelayerfeatures])
    return head #head_next_stage, head_parts




# input_img = Input(shape=(128, 128, 3))

# output_HG1 = create_hourglass_network(input_img, 1,64, bottleneck_block)

# model = Model(input_img,output_HG1)

# model.summary()
