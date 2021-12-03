import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
import cv2
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops
from Hourglass import *
from PIL import Image
import argparse
os.environ['CUDA_VISIBLE_DEVICES']='1'
def psnr(y_true,y_pred):
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

def ssim(img_1, img_2):
    c1  =  (0.01 * 255) **2
    c2  =  (0.03 * 255) **2
    img1 =  np.dot(img_1, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    img2=  np.dot(img_2, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq +mu2_sq + c1) * (sigma1_sq +sigma2_sq + c2))
    return ssim_map.mean()
def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                 ssims.append(ssim(img1,img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
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
    
    ##这里的conv kernel size变成了3 
    LR8 = Conv2D(3,3, padding='same', kernel_initializer='he_normal',name='C3_RESIDUAL')(upsample)
    LR8_ = add([LR8,upsample_img8],name='C3_RGB')
    # LR8 = PReLU()(LR8)
    # LR8 = Conv2D(3,3, padding='same')(LR8)
    out = []

    out.append(LR2_)
    out.append(LR4_)
    out.append(LR8_)
    
    return Model([input_img,upsample_img2,upsample_img4,upsample_img8], out)

def superFAN_Concat(FAN1=None,FAN2=None):
    input_img = Input(shape=(24,24,3,),name='C1_Input')
    upsample_img2 = Input(shape=(48,48,3,),name='C1_Upsample')
    upsample_img4 = Input(shape=(96,96,3,),name='C2_Upsample')
    upsample_img8 = Input(shape=(192,192,3,),name='C3_Upsample')
    
    upsample = generator(input_img)
    LR2 = Conv2D(3,3 , padding='same', kernel_initializer='he_normal',name='C1_RESIDUAL')(upsample)
    LR2_ = add([LR2,upsample_img2],name='C1_RGB')
   

    for index in range(3):
        upsample = res_block_gen( upsample, 3, 64, 1,"C2_%d"%index)
    LR2 = create_hourglass_network(1,64,bottleneck_block,8,FAN1)(LR2)
    upsample = Concatenate(name='C2_Concat')([upsample,LR2])
    upsample = BatchNormalization(name="C2_Encode_BN")(upsample)
    upsample = Activation('relu',name="C2_Encode_End")(upsample)
    
    
    upsample = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same",name="C2_Decode_End")(upsample)
    
    LR4 = Conv2D(3,3, padding='same', kernel_initializer='he_normal',name='C2_RESIDUAL')(upsample)
    LR4_ = add([LR4,upsample_img4],name='C2_RGB')
   

    for index in range(3):
        upsample = res_block_gen( upsample, 3, 64, 1,"C3_%d"%index)
    LR4 = create_hourglass_network(1,64,bottleneck_block,8,FAN2)(LR4)
    upsample = Concatenate(name='C3_Concat')([upsample,LR4])
    upsample = BatchNormalization(name="C3_Encode_BN")(upsample)
    upsample = Activation('relu',name="C3_Encode_End")(upsample)
    
    
    
    upsample = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same",name="C3_Decode_End")(upsample)
    
    upsample = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same",name="C3_conv1")(upsample)
    upsample = Activation('relu',name="C3_act1")(upsample)
    upsample = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same",name="C3_conv2")(upsample)
    upsample = Activation('relu',name="C3_act2")(upsample)
    
    ##这里的conv kernel size变成了3 
    LR8 = Conv2D(3,3, padding='same', kernel_initializer='he_normal',name='C3_RESIDUAL')(upsample)
    LR8_ = add([LR8,upsample_img8],name='C3_RGB')
    # LR8 = PReLU()(LR8)
    # LR8 = Conv2D(3,3, padding='same')(LR8)
    out = []
#     out.append(LR2)
#     out.append(LR4)
#     out.append(LR8)
    out.append(LR2_)
    out.append(LR4_)
    out.append(LR8_)
    
 
    return Model([input_img,upsample_img2,upsample_img4,upsample_img8], out)
def superFAN_Constrain(FAN1=None,FAN2=None,FAN3=None,SR=None):
    input_img = Input(shape=(24,24,3,),name='C1_Input')
    upsample_img2 = Input(shape=(48,48,3,),name='C1_Upsample')
    upsample_img4 = Input(shape=(96,96,3,),name='C2_Upsample')
    upsample_img8 = Input(shape=(192,192,3,),name='C3_Upsample')
    
    upsample = generator(input_img,SR)
    LR2 = Conv2D(3,3 , padding='same', kernel_initializer='he_normal',name='C1_RESIDUAL',weights=SR.get_layer('C1_RESIDUAL').get_weights())(upsample)
    LR2_ = add([LR2,upsample_img2],name='C1_RGB')
   

    for index in range(3):
        upsample = res_block_gen( upsample, 3, 64, 1,"C2_%d"%index,SR)
    LR2_FAN = create_hourglass_network(1,64,bottleneck_block,8,FAN1)(LR2)
    
    upsample = BatchNormalization(name="C2_Encode_BN",weights=SR.get_layer('C2_Encode_BN').get_weights())(upsample)
#     upsample = BatchNormalization(name="C2_Encode_BN")(upsample)
    upsample = Activation('relu',name="C2_Encode_End")(upsample)
    
    
    upsample = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same",name="C2_Decode_End",weights=SR.get_layer("C2_Decode_End").get_weights())(upsample)
#     upsample = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same",name="C2_Decode_End")(upsample)
    LR4 = Conv2D(3,3, padding='same', kernel_initializer='he_normal',name='C2_RESIDUAL',weights=SR.get_layer("C2_RESIDUAL").get_weights())(upsample)
    LR4_ = add([LR4,upsample_img4],name='C2_RGB')
   

    for index in range(3):
        upsample = res_block_gen( upsample, 3, 64, 1,"C3_%d"%index,SR)
    LR4_FAN = create_hourglass_network(1,64,bottleneck_block,8,FAN2)(LR4)
    
    upsample = BatchNormalization(name="C3_Encode_BN",weights=SR.get_layer('C3_Encode_BN').get_weights())(upsample)
#     upsample = BatchNormalization(name="C3_Encode_BN")(upsample)
    upsample = Activation('relu',name="C3_Encode_End")(upsample)
    
    
    
    upsample = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same",name="C3_Decode_End",weights=SR.get_layer("C3_Decode_End").get_weights())(upsample)
#     upsample = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same",name="C3_Decode_End")(upsample)
    upsample = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same",name="C3_conv1",weights=SR.get_layer('C3_conv1').get_weights())(upsample)
    upsample = Activation('relu',name="C3_act1")(upsample)
    upsample = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same",name="C3_conv2",weights=SR.get_layer('C3_conv2').get_weights())(upsample)
    upsample = Activation('relu',name="C3_act2")(upsample)
    
    ##这里的conv kernel size变成了3 
    LR8 = Conv2D(3,3, padding='same', kernel_initializer='he_normal',name='C3_RESIDUAL',weights=SR.get_layer('C3_RESIDUAL').get_weights())(upsample)
    LR8_ = add([LR8,upsample_img8],name='C3_RGB')
    LR8_FAN = create_hourglass_network(1,64,bottleneck_block,8,FAN3)(LR8)
    out = []
    out.append(LR2_)
    out.append(LR4_)
    out.append(LR8_)
    out.append(LR2_FAN)
    out.append(LR4_FAN)
    out.append(LR8_FAN)
    return Model([input_img,upsample_img2,upsample_img4,upsample_img8], out)

def predict(model,inputs,names):
    out = model.predict(inputs)
    size = out[0].shape[0]
    if not os.path.exists('X2'):
        os.mkdir('X2')
    if not os.path.exists('X4'):
        os.mkdir('X4')
    if not os.path.exists('X8'):
        os.mkdir('X8')
    for i in range(size):
        img48 = np.clip(out[0][i],0,255).astype(np.uint8)
        img96 = np.clip(out[1][i],0,255).astype(np.uint8)
        img192 = np.clip(out[2][i],0,255).astype(np.uint8)

        Image.fromarray(img48).save(os.path.join('X2',names[i]))
        Image.fromarray(img96).save(os.path.join('X4',names[i]))
        Image.fromarray(img192).save(os.path.join('X8',names[i]))

def test(model,inputs,targets,names):
    out = model.predict(inputs)
    size = targets[0].shape[0]
    ssim192 = []
    ssim96 = []
    ssim48 = []
    psnr192 = psnr(out[2],targets[2].astype(np.float32)).numpy()
    psnr96 = psnr(out[1],targets[1].astype(np.float32)).numpy()
    psnr48 = psnr(out[0],targets[0].astype(np.float32)).numpy()
    for i in range(size):
        ssim192.append(calculate_ssim(out[2][i],targets[2][i]))
        ssim96.append(calculate_ssim(out[1][i],targets[1][i]))
        ssim48.append(calculate_ssim(out[0][i],targets[0][i]))
    
    
    if not os.path.exists('X2'):
        os.mkdir('X2')
    if not os.path.exists('X4'):
        os.mkdir('X4')
    if not os.path.exists('X8'):
        os.mkdir('X8')
    
    for i in range(size):
        img48 = np.clip(out[0][i],0,255).astype(np.uint8)
        img96 = np.clip(out[1][i],0,255).astype(np.uint8)
        img192 = np.clip(out[2][i],0,255).astype(np.uint8)

        Image.fromarray(img48).save(os.path.join('X2',names[i]))
        Image.fromarray(img96).save(os.path.join('X4',names[i]))
        Image.fromarray(img192).save(os.path.join('X8',names[i]))
        
    '''
    list the result
    '''
    
    with open('result.txt','w') as file:
        file.write('Average psnr/ssim :\n')
        file.write('x2: %.2f/%.4f \n'%(sum(psnr48)/float(size),sum(ssim48)/float(size)))
        file.write('x4: %.2f/%.4f \n'%(sum(psnr96)/float(size),sum(ssim96)/float(size)))
        file.write('x8: %.2f/%.4f \n'%(sum(psnr192)/float(size),sum(ssim192)/float(size)))
        
        file.write('\n')
        
        for i in range(size):
            file.write('%s:\n'%names[i])
            file.write('x2: %.2f/%.4f \n'%(psnr48[i],ssim48[i]))
            file.write('x4: %.2f/%.4f \n'%(psnr96[i],ssim96[i]))
            file.write('x8: %.2f/%.4f \n'%(psnr192[i],ssim192[i]))
            file.write('\n')

def readDataset(path):
    Img_LR,Img_24_2_48,Img_24_2_96,Img_24_2_192,Img_192_2_48,Img_192_2_96,Img_HR= [],[],[],[],[],[],[]
    for image in os.listdir(path):
        HR = Image.open(os.path.join(path,image))
        img_192_2_96 = HR.resize([96,96],resample=Image.BICUBIC)
        img_192_2_48 = HR.resize([48,48],resample=Image.BICUBIC)
        LR = HR.resize([24,24],resample=Image.BICUBIC)
        
        img_24_2_48 = LR.resize([48,48],resample=Image.BICUBIC)
        img_24_2_96 = LR.resize([96,96],resample=Image.BICUBIC)
        img_24_2_192 = LR.resize([192,192],resample=Image.BICUBIC)
        
        Img_24_2_48.append(np.array(img_24_2_48))
        Img_24_2_96.append(np.array(img_24_2_96))
        Img_24_2_192.append(np.array(img_24_2_192))
        Img_LR.append(np.array(LR))
        
        Img_192_2_96.append(np.array(img_192_2_96))
        Img_192_2_48.append(np.array(img_192_2_48))
        Img_HR.append(np.array(HR))
    inputs = (np.array(Img_LR),np.array(Img_24_2_48),np.array(Img_24_2_96),np.array(Img_24_2_192))
    targets = (np.array(Img_192_2_48),np.array(Img_192_2_96),np.array(Img_HR))
    return inputs,targets


def readRealDataset(path):
    Img_LR, Img_24_2_48, Img_24_2_96, Img_24_2_192, Img_192_2_48, Img_192_2_96, Img_HR = [], [], [], [], [], [], []
    for image in os.listdir(path):
        LR = Image.open(os.path.join(path, image))

        img_24_2_48 = LR.resize([48, 48], resample=Image.BICUBIC)
        img_24_2_96 = LR.resize([96, 96], resample=Image.BICUBIC)
        img_24_2_192 = LR.resize([192, 192], resample=Image.BICUBIC)

        Img_24_2_48.append(np.array(img_24_2_48))
        Img_24_2_96.append(np.array(img_24_2_96))
        Img_24_2_192.append(np.array(img_24_2_192))
        Img_LR.append(np.array(LR))
    inputs = (np.array(Img_LR), np.array(Img_24_2_48), np.array(Img_24_2_96), np.array(Img_24_2_192))
    return inputs


class GAN():
    def __init__(self,lr):
        adam = tf.keras.optimizers.Adam(lr=lr)
        mse = tf.keras.losses.MeanSquaredError()
        Closs = tf.keras.losses.sparse_categorical_crossentropy
        model = superFAN_Concat(None,None)
        self.generator = model
#         self.generator.load_weights('../TEST/CFSR_new/050_shuffle.hdf5')
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=[Closs,'binary_crossentropy'],loss_weights=[1,1],
            optimizer=tf.keras.optimizers.Adam(lr=1e-2),
            metrics=['binary_accuracy'])
        
        identity ,valid= self.discriminator(self.generator.output[-1])
        self.combined = Model(self.generator.input, [self.generator.output[0],self.generator.output[1],
                                                     self.generator.output[2],identity,valid])



    def build_discriminator(self):
        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        df = 64
        hr_shape = (192,192,3)
        # Input img
        d0 = Input(shape=hr_shape)

        d1 = d_block(d0, df, bn=False)
        d2 = d_block(d1, df, strides=2)
        d3 = d_block(d2, df*2)
        d4 = d_block(d3, df*2, strides=2)
        d5 = d_block(d4, df*4)
        d6 = d_block(d5, df*4, strides=2)
        d7 = d_block(d6, df*8)
        d8 = d_block(d7, df*8, strides=2)

        d9 = Dense(df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        d11 = GlobalAveragePooling2D()(d10)
        identity = Dense(10064, activation='softmax')(d11)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, [identity,validity])

    def train(self,epochs):
            tfrecordlist = []
            for i in range(1,293):
                tfrecordlist.append('/home/fwb/WebFace/tfrecord_192/WebFace_train%d.tfrecords'%i)
            dataset = read_web_tfrecords(tfrecordlist,100)
            
            for epoch in range(epochs):
                iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
                dis_loss,g_loss = [],[]
                gradient_mse, gradient_dis = [],[]
                for i in tqdm(range(2919)):
                
                    inputs, targets = iterator.get_next()
                    
                    batch_size = inputs[0].shape[0]
    #                 print(targets[0].shape)
                    output = self.generator.predict(inputs)
                    real = np.ones((batch_size,12,12,1 ))
                    fake = np.zeros((batch_size,12,12,1))
                    

                    self.discriminator.trainable = True
                    # Train the discriminator (real classified as ones and generated as zeros)
                    d_loss_re= self.discriminator.train_on_batch(targets[2],[targets[-2],real])
                    d_loss_fa= self.discriminator.train_on_batch(output[2],[targets[-2],fake])
#                     print(d_loss_)
                    dis_loss.append((d_loss_re,d_loss_fa))
                    self.discriminator.trainable = False
                    # ---------------------
                    #  Train Generator
                    # ---------------------

                    # Train the generator (wants discriminator to mistake images as real)
#                     print(targets[-1][0].shape)
                    self.combined.train_on_batch(inputs, targets)
                    
                    g_loss.append(self.combined.train_on_batch(inputs, targets))
                    if batch_size!=100 :
                        break
#                     with tf.GradientTape() as gtape:
#                         gradient_mse.append(gtape.gradient(self.combined.output[2],self.combined.get_layer('C3_act2')))
#                     with tf.GradientTape() as gtape:
#                         gradient_dis.append(gtape.gradient(self.combined.output[-1],self.combined.get_layer('C3_act2')))
#                     print(self.combined.layers[-2].name)
                    a = np.random.randint(20)
                    plt.subplot(2,2,1)
                    plt.axis('off')
                    plt.imshow(output[2][a]/255.)
                    plt.subplot(2,2,2)
                    plt.axis('off')
                    plt.imshow(targets[2][a]/255.)
                    a = np.random.randint(20)
                    plt.subplot(2,2,3) 
                    plt.axis('off')
                    plt.imshow(output[2][a]/255.)
                    plt.subplot(2,2,4)
                    plt.axis('off')
                    plt.imshow(targets[2][a]/255.)
                    plt.show()
                clear_output()
                self.combined.evaluate(inputs, targets)
#                 for i in g_loss:
#                     print(i)
    #             print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, loss1, 100*acc, loss2))
                self.discriminator.trainable = True
                self.combined.save_weights('GAN/model{:03d}_TEST.hdf5'.format(epoch+1))
                a = np.random.randint(20)
                plt.subplot(2,2,1)
                plt.axis('off')
                plt.imshow(output[2][a]/255.)
                plt.subplot(2,2,2)
                plt.axis('off')
                plt.imshow(targets[2][a]/255.)
                a = np.random.randint(20)
                plt.subplot(2,2,3) 
                plt.axis('off')
                plt.imshow(output[2][a]/255.)
                plt.subplot(2,2,4)
                plt.axis('off')
                plt.imshow(targets[2][a]/255.)
                plt.show()
                np.save('GAN/dis_loss{:03d}.npy'.format(epoch+1),dis_loss)
                np.save('GAN/g_loss{:03d}.npy'.format(epoch+1),g_loss)
                np.save('GAN/gradient_mse{:03d}.npy'.format(epoch+1),gradient_mse)
                np.save('GAN/gradient_dis{:03d}.npy'.format(epoch+1),gradient_dis)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="images path")
    parser.add_argument("-real",action="store_true")
    parser.add_argument("--model", help="model name",choices=['Baseline','B+Shape','B+FaceNet','CSRNet','GAN'])
    args = parser.parse_args()
    if args.path:
        if(args.real):
            inputs = readRealDataset(args.path)
        else:
            inputs,targets = readDataset(args.path)
        path = args.path
    else:
        if (args.real):
            inputs = readRealDataset('./example')
        else:
            inputs,targets = readDataset('./example')
        path = './example'
    if args.model:
        if (args.model=='Baseline'):
            model = superFAN()
            model.load_weights('model/base.hdf5')
        elif (args.model=='B+Shape'):
            model = superFAN_Concat()
            model.load_weights('model/asinput.hdf5')
        elif (args.model=='B+FaceNet'):
            model = superFAN()
            model.load_weights('model/facenet.hdf5')
        elif (args.model=='B+EfficientNet'):
            model = superFAN()
            model.load_weights('model/EfficientNet.hdf5')
        elif (args.model=='B+Identity'):
            model = superFAN()
            model.load_weights('model/facenet.hdf5')
        elif (args.model=='GAN'):
            model = GAN(1e-3).combined
            model.load_weights('E:/lzl/GAN_w/model001_TEST_0040.hdf5')
        else:
            model = superFAN_Concat()
            model.load_weights('model/final.hdf5')
    else:
        model = superFAN_Concat()
        model.load_weights('model/final.hdf5')
    if args.real:
        predict(model,inputs,os.listdir(path))
    else:
        test(model,inputs,targets,os.listdir(path))
    
    
    

