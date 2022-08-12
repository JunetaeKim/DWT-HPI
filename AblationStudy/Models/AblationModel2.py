from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import os

import numpy as np
import math

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Activation,Dropout
from tensorflow.keras import backend as K,losses
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K,losses


class LossHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self,logs={}):
        self.losses=[]
        self.val_losses=[]
        
    def on_epoch_end(self,batch,logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

   
    
class DoGenVec(tf.keras.layers.Layer):
    
    def __init__(self, OutDimes):
        super(DoGenVec, self).__init__()
        self.Shapelet1Size = OutDimes[0]
        self.WinSize = OutDimes[1]
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'Shapelet1Size': self.Shapelet1Size,
            'WinSize': self.WinSize})
        return config
    
    def build(self, input_shape):
        self.GenVec = self.add_weight("GenVec", shape=[self.Shapelet1Size, self.WinSize], initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01))
    
    def call(self, input):
        input = K.sum(input) * 0 + 1 
        return (input*self.GenVec)
    
    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.98
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))     


if __name__ == "__main__":
    
    BatchSize = 800

    TrainInp = np.load('../ProcessedData/ABPInp_Train.npy',allow_pickle=True)
    TrainOut = np.load('../ProcessedData/HypoOut_Train.npy',allow_pickle=True)
    ValInp = np.load('../ProcessedData/ABPInp_Val.npy',allow_pickle=True)
    ValOut = np.load('../ProcessedData/HypoOut_Val.npy',allow_pickle=True)
    
    strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) 
    with strategy.scope():

        Shapelet1Size = 30
        PaddSideForSim = 12
        FrameSize = 50
        AttSize = 5

        InputVec = Input(shape=(9000), name='Input')

        FC1 = Conv1D(filters= 1,kernel_size= 20, strides=1, activation='softplus',padding="valid")(InputVec[:,:,None])
        FC1 = MaxPooling1D(pool_size=2,strides=2)(FC1)
        FC2 = Conv1D(filters= 1,kernel_size= 20, strides=1, activation='softplus',padding="valid")(FC1)
        FC2 = MaxPooling1D(pool_size=2,strides=2)(FC2)
        FC3 = Conv1D(filters= 1,kernel_size= 20, strides=1, activation='softplus',padding="valid")(FC2)
        FC3 = MaxPooling1D(pool_size=2,strides=2)(FC3)
        FC4 = Conv1D(filters= 1,kernel_size= 20, strides=1, activation='softplus',padding="valid")(FC3)
        FC4 = MaxPooling1D(pool_size=2,strides=2)(FC4)
        LP5_Down = Dense(232, activation='sigmoid')(FC4[:,:,0])

        LP5_FeatDim = tf.signal.frame(LP5_Down, FrameSize, 1)
        
        
        ### Shape Generation 
        GenVecLayer = DoGenVec([Shapelet1Size, FrameSize])
        GenVec = Activation('sigmoid')(GenVecLayer(InputVec)) 

        LP5_X_sqare = K.sum(K.square(LP5_FeatDim), axis=2,keepdims=True)
        LP5_Y_sqare = K.sum(K.square(GenVec[:]), axis=1)[None,None]
        LP5_XY = tf.matmul(LP5_FeatDim, GenVec[:], transpose_b=True)
        LP5_Dist = (LP5_X_sqare + LP5_Y_sqare - 2*LP5_XY) 
        LP5_Dist_sqrt = K.sqrt(LP5_Dist+K.epsilon())

        ### Cluster Loss
        ClLoss = K.mean(K.min(LP5_Dist, axis=1))

        ### Features
        Features_W = K.exp(-LP5_Dist_sqrt) 
        Features = K.max(Features_W, axis=1) 
        BinOut = Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=0),name='BinOut')(Dropout(0.0)(Features))

        BinModel = Model(InputVec, BinOut)
        BinModel.add_loss(ClLoss)
        BinModel.add_metric(ClLoss, name='ClLoss')

        lrate = 0.0005
        decay = 1e-6
        adam = tf.keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=decay)
        BinModel.compile(loss='binary_crossentropy', optimizer='adam',  metrics={'BinOut':['binary_crossentropy','AUC']}, ) #'binary_crossentropy' 

        # Run
        SaveFilePath = './Logs/SaveAblation2_{epoch:d}_{binary_crossentropy:.5f}_{auc:.5f}_{ClLoss:.5f}_{val_binary_crossentropy:.5f}_{val_auc:.5f}_{val_ClLoss:.5f}.hdf5'
        checkpoint = ModelCheckpoint(SaveFilePath,monitor=('val_loss'),verbose=0, save_best_only=True, mode='auto' ,period=1) 
        earlystopper = EarlyStopping(monitor='val_loss', patience=600, verbose=1,restore_best_weights=True)
        history = LossHistory()

        BinModel.fit(TrainInp, TrainOut, validation_data = (ValInp,ValOut), verbose=1, epochs=5000, callbacks=[history,earlystopper,checkpoint], )
        SaveFilePath = './Logs/ShapeBin_Ablation2_end.hdf5'
        BinModel.save(SaveFilePath)  