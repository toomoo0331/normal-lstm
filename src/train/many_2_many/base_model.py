import tensorflow as tf
import keras
from keras.layers import Input ,Dense, Dropout, Activation, LSTM, Bidirectional
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape, BatchNormalization, Add, MaxPooling1D
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils import np_utils
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import Sequence
from keras.initializers import glorot_normal,orthogonal

from many_2_many import resnet18

class Prediction :
    def __init__(self, timesteps, n_out, width, height):
        self.maxlen = 2*timesteps+1
        self.n_out = n_out
        self.width=width
        self.height=height
        self.resnet = resnet18.Resnet18(width, height, 3).get_resnet18()
        self.sharedLSTMmodel = self.create_sharedLSTMmodel()
        self.sharedRateModel = self.create_sharedRatemodel()
        self.sharedAutoRegModel = self.create_sharedAutoRegLSTMmodel()
    
    
    def create_sharedLSTMmodel(self):
        inputs = Input(shape=(self.maxlen, self.height, self.width, 3))
        
        x = TimeDistributed(self.resnet,name="resnet")(inputs)
        x = TimeDistributed(GlobalAveragePooling2D(), name="GAP")(x)
        x = TimeDistributed(Dense(512,activation='relu'), name="dense")(x)
        predictions = Bidirectional(LSTM(128, batch_input_shape = (None, self.maxlen, 512),
             kernel_initializer = glorot_normal(seed=20181020),
             recurrent_initializer = orthogonal(gain=1.0, seed=20181020), 
             dropout = 0.01, 
             recurrent_dropout = 0.01))(x)
        predictions =Reshape((1,256))(predictions)
        shared_layers = Model(inputs, predictions, name="shared_LSTMlayers")
        return shared_layers
    
    
    def create_sharedRatemodel(self):
        inputs = Input(shape=(512,))
        mid_dense = Dense(256, activation='relu')(inputs)
        predictions = Dense(self.maxlen, activation='sigmoid')(mid_dense)
        shared_layers = Model(inputs, predictions, name="shared_Ratelayers")
        return shared_layers
    
    def create_sharedAutoRegLSTMmodel(self):
        inputs = Input(shape=(2*self.maxlen,1))
        
        mid_lstm = Bidirectional(LSTM(128, batch_input_shape = (None, 2*self.maxlen,1),
             kernel_initializer = glorot_normal(seed=20181020),
             recurrent_initializer = orthogonal(gain=1.0, seed=20181020), 
             dropout = 0.01, 
             recurrent_dropout = 0.01,
             return_sequences=True))(inputs)

        mid_dense = Dense(256, activation='relu')(mid_lstm)
        predictions = Dense(self.maxlen, activation='sigmoid')(mid_dense)

        shared_layers = Model(inputs, predictions, name="shared_AutoRegLSTMlayers")
        return shared_layers


    def create_model(self):
        before1 = Input(shape=(self.maxlen,))
        before2 = Input(shape=(self.maxlen,))
        before3 = Input(shape=(self.maxlen,))
        before4 = Input(shape=(self.maxlen,))
        before5 = Input(shape=(self.maxlen,))

        model_input1 = Input(shape=(self.maxlen, self.height, self.width, 3))
        model_input2 = Input(shape=(self.maxlen, self.height, self.width, 3))
        model_input3 = Input(shape=(self.maxlen, self.height, self.width, 3))
        model_input4 = Input(shape=(self.maxlen, self.height, self.width, 3))
        model_input5 = Input(shape=(self.maxlen, self.height, self.width, 3))
        
        mid_feature1 = self.sharedLSTMmodel(model_input1)
        mid_feature2 = self.sharedLSTMmodel(model_input2)
        mid_feature3 = self.sharedLSTMmodel(model_input3)
        mid_feature4 = self.sharedLSTMmodel(model_input4)
        mid_feature5 = self.sharedLSTMmodel(model_input5)
        
        
        pooling_vector = keras.layers.concatenate([mid_feature1, mid_feature2, mid_feature3, mid_feature4, mid_feature5], axis=1)
        pooling_vector = MaxPooling1D(pool_size=self.n_out)(pooling_vector)
        
        merge_feature1 = keras.layers.concatenate([mid_feature1, pooling_vector], axis=-1)
        merge_feature2 = keras.layers.concatenate([mid_feature2, pooling_vector], axis=-1)
        merge_feature3 = keras.layers.concatenate([mid_feature3, pooling_vector], axis=-1)
        merge_feature4 = keras.layers.concatenate([mid_feature4, pooling_vector], axis=-1)
        merge_feature5 = keras.layers.concatenate([mid_feature5, pooling_vector], axis=-1)
        
        merge_feature1 = Reshape((512,))(merge_feature1)
        merge_feature2 = Reshape((512,))(merge_feature2)
        merge_feature3 = Reshape((512,))(merge_feature3)
        merge_feature4 = Reshape((512,))(merge_feature4)
        merge_feature5 = Reshape((512,))(merge_feature5)
        
        predictions1 = self.sharedRateModel(merge_feature1)
        predictions2 = self.sharedRateModel(merge_feature2)
        predictions3 = self.sharedRateModel(merge_feature3)
        predictions4 = self.sharedRateModel(merge_feature4)
        predictions5 = self.sharedRateModel(merge_feature5)


        merge_pred1 = keras.layers.concatenate([before1, predictions1], axis=-1)
        merge_pred2 = keras.layers.concatenate([before2, predictions2], axis=-1)
        merge_pred3 = keras.layers.concatenate([before3, predictions3], axis=-1)
        merge_pred4 = keras.layers.concatenate([before4, predictions4], axis=-1)
        merge_pred5 = keras.layers.concatenate([before5, predictions5], axis=-1)


        merge_pred1 = Reshape((2*self.maxlen,1))(merge_pred1)
        merge_pred2 = Reshape((2*self.maxlen,1))(merge_pred2)
        merge_pred3 = Reshape((2*self.maxlen,1))(merge_pred3)
        merge_pred4 = Reshape((2*self.maxlen,1))(merge_pred4)
        merge_pred5 = Reshape((2*self.maxlen,1))(merge_pred5)

        final_predictions1 = self.sharedAutoRegModel(merge_pred1)
        final_predictions2 = self.sharedAutoRegModel(merge_pred2)
        final_predictions3 = self.sharedAutoRegModel(merge_pred3)
        final_predictions4 = self.sharedAutoRegModel(merge_pred4)
        final_predictions5 = self.sharedAutoRegModel(merge_pred5)


        model = Model(inputs=[before1, before2, before3, before4, before5, model_input1, model_input2, model_input3, model_input4, model_input5], 
                      outputs=[final_predictions1, final_predictions2, final_predictions3, final_predictions4, final_predictions5])
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model