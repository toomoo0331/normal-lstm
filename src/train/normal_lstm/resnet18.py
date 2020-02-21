import tensorflow as tf
import keras
from keras.layers import Input, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Add
from keras.models import Model


class Resnet18 :
    ##
    ##　Resnet-18の作成
    ##
    def __init__(self, width, height, channels):
        self.img_rows=height
        self.img_cols=width
        self.img_channels=channels
    
    def rescell(self, data, filters, kernel_size, option=False):
        strides=(1,1)
        if option:
            strides=(2,2)
        x=Convolution2D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same")(data)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        
        data=Convolution2D(filters=int(x.shape[3]), kernel_size=(1,1), strides=strides, padding="same")(data)
        x=Convolution2D(filters=filters,kernel_size=kernel_size,strides=(1,1),padding="same")(x)
        x=BatchNormalization()(x)
        x=Add()([x,data])
        x=Activation('relu')(x)
        
        return x
    
    def create_resnet18(self):
        inputs=Input(shape=(self.img_rows, self.img_cols, self.img_channels))
        x=Convolution2D(64,(7,7), padding="same", input_shape=(self.img_rows, self.img_cols),activation="relu")(inputs)
        x=MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        x=self.rescell(x,64,(3,3))
        x=self.rescell(x,64,(3,3))
        x=self.rescell(x,64,(3,3))
        
        x=self.rescell(x,128,(3,3),True)
        x=self.rescell(x,128,(3,3))

        x=self.rescell(x,256,(3,3),True)
        x=self.rescell(x,256,(3,3))
        
        x=self.rescell(x,512,(3,3),True)
        x=self.rescell(x,512,(3,3))
        
        model=Model(inputs=inputs,outputs=[x])
        return model

    def get_resnet18(self):
        return self.create_resnet18()
