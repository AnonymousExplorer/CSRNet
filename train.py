import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
import numpy as np
import math
import pickle
import cv2
from mtcnn.mtcnn import MTCNN
#from scheduler import *
from  tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from Hourglass import *

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.ops import math_ops
from inception_resnet_v1_tf2 import *
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x), axis=axis, keepdims=True), epsilon))
    return output


def perceptual_loss(y_true,y_pred):
    emd_c3 = l2_normalize(extractor(y_pred))
    emd_gt = l2_normalize(extractor(y_true))
    return K.sqrt(K.sum(K.square(emd_c3 - emd_gt), axis=-1))
   
def perceptual_loss_96(y_true,y_pred):
    emd_c3 = l2_normalize(extractor_96(y_pred))
    emd_gt = l2_normalize(extractor_96(y_true))
    return K.sqrt(K.sum(K.square(emd_c3 - emd_gt), axis=-1))

def Accuracy(y_true,y_pred):
    if len(y_true.shape)>2:
        return 0
    return sparse_accuracy(y_true,y_pred)
def Eff_train(size):

    extractor = tf.keras.applications.EfficientNetB1(include_top=False, weights="imagenet",input_shape=(size,size,3))
    x = GlobalAveragePooling2D()(extractor.output)
    x = Dropout(0.2)(x)
    x = Dense(512, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Dense(10064, activation='softmax')(x)
    Efficient = Model(inputs=extractor.input, outputs=x)
    
    return Efficient


def generate_gt(size, landmark_list, sigma = 1):
    '''
    return N * H * W
    '''
    cnt = np.zeros(size)
    for l in landmark_list:
        cnt += _generate_one_heatmap(size,l,sigma)
    return cnt

def _generate_one_heatmap(size, landmark, sigma):
    w, h = size
    x_range = np.arange(start=0, stop=w, dtype=int)
    y_range = np.arange(start=0, stop=h, dtype=int)
    xx, yy = np.meshgrid(x_range, y_range)
    d2 = (xx - landmark[0])**2 + (yy - landmark[1])**2
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = tf.math.exp(-exponent)
    return heatmap
def get_heatmap(pred,size):
        

    new_pred = list()
    new_pred.append(pred[11])
    new_pred.append(pred[2])
    new_pred.append(pred[0])
    new_pred.append(pred[18])
    new_pred.append(pred[27])
    landmark_skin = new_pred

    new_pred = list()
    new_pred.append(pred[43])
    new_pred.append(pred[49])
    new_pred.append(pred[50])
    new_pred.append(pred[46])
    new_pred.append(pred[45])
    landmark_l_brow = new_pred

    new_pred = list()
    new_pred.append(pred[102])
    new_pred.append(pred[104])
    new_pred.append(pred[101])
    new_pred.append(pred[99])
    new_pred.append(pred[97])
    landmark_r_brow = new_pred

    new_pred = list()
    new_pred.append(pred[35])
    new_pred.append(pred[40])
    new_pred.append(pred[33])
    new_pred.append(pred[39])
    new_pred.append(pred[38])
    landmark_l_eye = new_pred

    new_pred = list()
    new_pred.append(pred[89])
    new_pred.append(pred[87])
    new_pred.append(pred[93])
    new_pred.append(pred[94])
    new_pred.append(pred[92])
    landmark_r_eye = new_pred

    new_pred = list()
    new_pred.append(pred[72])
    new_pred.append(pred[77])
    new_pred.append(pred[86])
    new_pred.append(pred[83])
    new_pred.append(pred[80])
    landmark_nose = new_pred

    new_pred = list()
    new_pred.append(pred[52])
    new_pred.append(pred[63])
    new_pred.append(pred[67])
    new_pred.append(pred[61])
    new_pred.append(pred[53])
    landmark_outer_mouth = new_pred

    new_pred = list()
    new_pred.append(pred[65])
    new_pred.append(pred[66])
    new_pred.append(pred[70])
    new_pred.append(pred[69])
    new_pred.append(pred[60])
    landmark_inner_mouth = new_pred




    # generate heatmap
    landmarks = (generate_gt(size,landmark_l_brow),generate_gt(size,landmark_r_brow),generate_gt(size,landmark_l_eye),
                 generate_gt(size,landmark_r_eye),generate_gt(size,landmark_nose),generate_gt(size,landmark_outer_mouth),
                 generate_gt(size,landmark_inner_mouth),generate_gt(size,landmark_skin))
    heat_maps = tf.stack(landmarks,2)
    print(heat_maps)
    return heat_maps
def res_block_gen(model, kernal_size, filters, strides,name,SR=None) :
    
    gen = model
    if SR:
        model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same",name=name+'_1conv',weights=SR.get_layer(name+'_1conv').get_weights())(model)
        model = BatchNormalization(name=name+'_1bn',weights=SR.get_layer(name+'_1bn').get_weights())(model)
    else:
        model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same",name=name+'_1conv')(model)
        model = BatchNormalization(name=name+'_1bn')(model)
    model = Activation('relu',name=name+'_act')(model)
    if SR:
        model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same",name=name+'_2conv',weights=SR.get_layer(name+'_2conv').get_weights())(model)
        model = BatchNormalization(name=name+'_2bn',weights=SR.get_layer(name+'_2bn').get_weights())(model)
    else:
        model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same",name=name+'_2conv')(model)
        model = BatchNormalization(name=name+'_2bn')(model)
    model = add([gen, model],name=name+'_add')
    
    return model

def generator(gen_input,SR=None):
        
#     gen_input = Input(shape = noise_shape)
    if SR:
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same",name="C1_feature_extract",weights=SR.get_layer("C1_feature_extract").get_weights())(gen_input)
    else:
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same",name="C1_feature_extract")(gen_input)
    model = Activation('relu',name="C1_RELU")(model)
    
    for index in range(12):
        model= res_block_gen(model, 3, 64, 1,"C1_%d"%index,SR)
    
    
    model = BatchNormalization(name="C1_Encode_BN")(model)
    model = Activation('relu',name="C1_Encode_End")(model)
    model = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same",name="C1_Decode_End")(model)
    return model

def psnr(y_true,y_pred):
    if(len(y_true.shape)<=2):
        return 0 
    if(y_pred.shape[3] == 5):
        return 0
    img1 = tf.matmul(y_true, [[65.481], [128.553], [24.966]]) / 255.0 + 16.0
    img2 = tf.matmul(y_pred, [[65.481], [128.553], [24.966]]) / 255.0 + 16.0
    mse = math_ops.reduce_mean(math_ops.squared_difference(img1,img2), [-3, -2, -1])
    def log10(x):
        numerator = tf.compat.v1.log(x)
        denominator = tf.compat.v1.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    return 10*log10(255.0*255.0/mse)

def superFAN():

    input_img = Input(shape=(24,24,3,),name='C1_Input')
    upsample_img2 = Input(shape=(48,48,3,),name='C1_Upsample')
    upsample_img4 = Input(shape=(96,96,3,),name='C2_Upsample')
    upsample_img8 = Input(shape=(192,192,3,),name='C3_Upsample')
    
    upsample = generator(input_img)
    LR2 = Conv2D(3,3 , padding='same', kernel_initializer='he_normal',name='C1_RESIDUAL')(upsample)
    LR2_ = add([LR2,upsample_img2],name='C1_RGB')
   

    for index in range(3):
        upsample = res_block_gen( upsample, 3, 64, 1,"C2_%d"%index)
    upsample = BatchNormalization(name="C2_Encode_BN")(upsample)
    upsample = Activation('relu',name="C2_Encode_End")(upsample)
    upsample = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same",name="C2_Decode_End")(upsample)
    
    LR4 = Conv2D(3,3, padding='same', kernel_initializer='he_normal',name='C2_RESIDUAL')(upsample)
    LR4_ = add([LR4,upsample_img4],name='C2_RGB')
   

    for index in range(3):
        upsample = res_block_gen( upsample, 3, 64, 1,"C3_%d"%index)
    upsample = BatchNormalization(name="C3_Encode_BN")(upsample)
    upsample = Activation('relu',name="C3_Encode_End")(upsample)
    upsample = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same",name="C3_Decode_End")(upsample)
    
    upsample = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same",name="C3_conv1")(upsample)
    upsample = Activation('relu',name="C3_act1")(upsample)
    upsample = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same",name="C3_conv2")(upsample)
    upsample = Activation('relu',name="C3_act2")(upsample)

    LR8 = Conv2D(3,3, padding='same', kernel_initializer='he_normal',name='C3_RESIDUAL')(upsample)
    LR8_ = add([LR8,upsample_img8],name='C3_RGB')

    out = []

    out.append(LR2_)
    out.append(LR4_)
    out.append(LR8_)
 
    return Model([input_img,upsample_img2,upsample_img4,upsample_img8], out)
def superFAN_C1():
    model = superFAN()
    for layer in model.layers:
        if layer.name.split('_')[0] != 'C1':
            layer.trainable = False
    return model
def superFAN_C1_C2(train_C1):
    if train_C1:
        model = superFAN()
        for layer in model.layers:
            if layer.name.split('_')[0] == 'C3':
                layer.trainable = False
    else:
        model = superFAN()
        for layer in model.layers:
            if layer.name.split('_')[0] == 'C1' or layer.name.split('_')[0] =='C3' :
                layer.trainable = False
    return model
def superFAN_C1_C2_C3(train_C1_C2):
    if train_C1_C2:
        model = superFAN()
    else:
        model = superFAN()
        for layer in model.layers:
            if layer.name.split('_')[0] == 'C1' or layer.name.split('_')[0] =='C2':
                layer.trainable = False
    return model
def superFAN_Concat(FAN1=None,FAN2=None,SR=None):
    input_img = Input(shape=(24,24,3,),name='C1_Input')
    upsample_img2 = Input(shape=(48,48,3,),name='C1_Upsample')
    upsample_img4 = Input(shape=(96,96,3,),name='C2_Upsample')
    upsample_img8 = Input(shape=(192,192,3,),name='C3_Upsample')
    
    upsample = generator(input_img,SR)
    LR2 = Conv2D(3,3 , padding='same', kernel_initializer='he_normal',name='C1_RESIDUAL',weights=SR.get_layer('C1_RESIDUAL').get_weights())(upsample)
    LR2_ = add([LR2,upsample_img2],name='C1_RGB')
   

    for index in range(3):
        upsample = res_block_gen( upsample, 3, 64, 1,"C2_%d"%index,SR)
    LR2 = create_hourglass_network(1,64,bottleneck_block,8,FAN1)(LR2)
    upsample = Concatenate(name='C2_Concat')([upsample,LR2])
    upsample = BatchNormalization(name="C2_Encode_BN")(upsample)
    upsample = Activation('relu',name="C2_Encode_End")(upsample)

    upsample = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same",name="C2_Decode_End")(upsample)
    LR4 = Conv2D(3,3, padding='same', kernel_initializer='he_normal',name='C2_RESIDUAL',weights=SR.get_layer("C2_RESIDUAL").get_weights())(upsample)
    LR4_ = add([LR4,upsample_img4],name='C2_RGB')
   

    for index in range(3):
        upsample = res_block_gen( upsample, 3, 64, 1,"C3_%d"%index,SR)
    LR4 = create_hourglass_network(1,64,bottleneck_block,8,FAN2)(LR4)
    upsample = Concatenate(name='C3_Concat')([upsample,LR4])
    upsample = BatchNormalization(name="C3_Encode_BN")(upsample)
    upsample = Activation('relu',name="C3_Encode_End")(upsample)
    
    


    upsample = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same",name="C3_Decode_End")(upsample)
    upsample = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same",name="C3_conv1",weights=SR.get_layer('C3_conv1').get_weights())(upsample)
    upsample = Activation('relu',name="C3_act1")(upsample)
    upsample = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same",name="C3_conv2",weights=SR.get_layer('C3_conv2').get_weights())(upsample)
    upsample = Activation('relu',name="C3_act2")(upsample)

    LR8 = Conv2D(3,3, padding='same', kernel_initializer='he_normal',name='C3_RESIDUAL',weights=SR.get_layer('C3_RESIDUAL').get_weights())(upsample)
    LR8_ = add([LR8,upsample_img8],name='C3_RGB')

    out = []
  
    out.append(LR4_)
    out.append(LR8_)
    out.append(LR2)
    out.append(LR4)
    out.append(LR2_)
    out.append(LR4_)
    out.append(LR8_)
 
    return Model([input_img,upsample_img2,upsample_img4,upsample_img8], out)
def tfrecord_to_tensor(example):
    features = tf.io.parse_single_example(
        example,
        features={
            'id_info': tf.io.FixedLenFeature((), tf.int64),
            'target': tf.io.FixedLenFeature([], tf.string),
            'img_192_2_24': tf.io.FixedLenFeature([], tf.string),
            'img_192_2_48': tf.io.FixedLenFeature([], tf.string),
            'img_192_2_96': tf.io.FixedLenFeature([], tf.string),
            'img_24_2_48': tf.io.FixedLenFeature([], tf.string),
            'img_24_2_96': tf.io.FixedLenFeature([], tf.string),
            'img_24_2_192': tf.io.FixedLenFeature([], tf.string),
            'landmark_192': tf.io.FixedLenFeature([], tf.string)
        })

    id_info = features['id_info']
#     name = features['name']

    target = features['target']
    target = tf.io.decode_raw(target, tf.uint8)
    target = tf.reshape(target, [192,192,3])

    img_192_2_24 = features['img_192_2_24']
    img_192_2_24 = tf.io.decode_raw(img_192_2_24, tf.uint8)
    img_192_2_24 = tf.reshape(img_192_2_24, [24,24,3])

    img_192_2_48 = features['img_192_2_48']
    img_192_2_48 = tf.io.decode_raw(img_192_2_48,tf.uint8)
    img_192_2_48 = tf.reshape(img_192_2_48,[48,48,3])

    img_192_2_96 = features['img_192_2_96']
    img_192_2_96 = tf.io.decode_raw(img_192_2_96,tf.uint8)
    img_192_2_96 = tf.reshape(img_192_2_96,[96,96,3])

    img_24_2_48 = features['img_24_2_48']
    img_24_2_48 = tf.io.decode_raw(img_24_2_48,tf.uint8)
    img_24_2_48 = tf.reshape(img_24_2_48,[48,48,3])

    img_24_2_96 = features['img_24_2_96']
    img_24_2_96 = tf.io.decode_raw(img_24_2_96,tf.uint8)
    img_24_2_96 = tf.reshape(img_24_2_96,[96,96,3])

    img_24_2_192 = features['img_24_2_192']
    img_24_2_192 = tf.io.decode_raw(img_24_2_192,tf.uint8)
    img_24_2_192 = tf.reshape(img_24_2_192,[192,192,3])

 
    landmark_192 = features['landmark_192']
    landmark_192 = tf.io.decode_raw(landmark_192,tf.float32)
    landmark_192 = tf.reshape(landmark_192,[106,2])
    
    target = tf.cast(target,tf.float32)
    img_192_2_24 = tf.cast(img_192_2_24,tf.float32)
    img_192_2_48 = tf.cast(img_192_2_48,tf.float32)
    img_192_2_96 = tf.cast(img_192_2_96,tf.float32)
    img_24_2_48 = tf.cast(img_24_2_48,tf.float32)
    img_24_2_96 = tf.cast(img_24_2_96,tf.float32)
    img_24_2_192 = tf.cast(img_24_2_192,tf.float32)
   
    landmark_96 = landmark_192/2.0
    landmark_48 = landmark_192/4.0

    heatmap_96 = get_heatmap(landmark_96,(96,96))
    heatmap_48 = get_heatmap(landmark_48,(48,48))

    inputs = (img_192_2_24,img_24_2_48,img_24_2_96,img_24_2_192)
    targets = (img_192_2_96,target,heatmap_48,heatmap_96,img_192_2_48,img_192_2_96,target)



    return inputs,targets


def read_web_tfrecords(save_path,size):
    dataset = tf.data.TFRecordDataset(save_path)
    dataset = dataset.map(tfrecord_to_tensor,tf.data.experimental.AUTOTUNE).shuffle(12000).batch(size)
    return dataset

def psnr(y_true,y_pred):
    if(len(y_true.shape)<=2):
        return 0 
    if(y_pred.shape[3] == 8):
        return 0
    img1 = tf.matmul(y_true, [[65.481], [128.553], [24.966]]) / 255.0 + 16.0
    img2 = tf.matmul(y_pred, [[65.481], [128.553], [24.966]]) / 255.0 + 16.0
    mse = math_ops.reduce_mean(math_ops.squared_difference(img1,img2), [-3, -2, -1])
    def log10(x):
        numerator = tf.compat.v1.log(x)
        denominator = tf.compat.v1.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    return 10*log10(255.0*255.0/mse)


global sparse_accuracy
sparse_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

crop_size = 192
global extractor
extractor = InceptionResNetV1(input_shape=(crop_size, crop_size, 3),weights_path='model_005602_TF2.h5')
extractor.trainable = False

crop_size = 96
global extractor_96
extractor_96 = InceptionResNetV1(input_shape=(crop_size, crop_size, 3),weights_path='model_005602_TF2.h5')
extractor_96.trainable = False
SRModel = superFAN()
model = superFAN_Concat(None,None,SRModel)


adam = tf.keras.optimizers.Adam(lr=5e-4)
mse = tf.keras.losses.MeanSquaredError()
Closs = tf.keras.losses.sparse_categorical_crossentropy
model.compile(optimizer=adam,loss=[perceptual_loss_96,perceptual_loss,mse,mse,mse,mse,mse],metrics=[psnr])
tfrecordlist = []
tmp = []
for i in range(1,3):
    tmp.append(i)
for i in tmp:
    tfrecordlist.append('./tfrecord_192/WebFace_train%d.tfrecords'%i)
dataset = read_web_tfrecords(tfrecordlist,64)
history = model.fit(dataset, epochs=1,callbacks=[TensorBoard(log_dir='log')])
with open(exp+'/'+'result.npy', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
