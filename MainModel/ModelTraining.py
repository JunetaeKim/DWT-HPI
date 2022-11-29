from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Activation,Dropout
from tensorflow.keras import backend as K,losses
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam

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


# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.98
# Create a session with the above options specified.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


b = 0.08 # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
N = int(np.ceil((4 / b)))
if not N % 2:  # Make sure that N is odd.
    N += 1  
RanVar = tf.constant(np.arange(N), dtype=tf.float32)

# Intergration of duplicated functions in ModelTraining.py and Main Result-SIM.ipynb
def FilterGen (FC):
    # Sinc function
    X = (2 * FC * (RanVar[None] - (N - 1) / 2))
    X = tf.where(X == 0, K.epsilon(), X)
    SinF = tf.sin(np.pi*X)/(np.pi*X)

    # Black man window
    BW = 0.42 - 0.5 * tf.math.cos(2 * np.pi * RanVar / (N - 1)) +  0.08 * tf.math.cos(4 * np.pi * RanVar / (N - 1))

    SinFBW = SinF * BW
    LP = SinFBW /  K.sum(SinFBW, axis=-1, keepdims=True)
    HP = -LP
    TmpZeros = tf.zeros((N - 1) // 2)
    TmpOnes = tf.ones(1)
    AddOne = tf.concat([TmpZeros,TmpOnes,TmpZeros], axis=0)
    HP += AddOne

    return LP, HP

def DownSampling (ToDown):
    if ToDown.shape[1] % 2 != 0: 
        ToDown = tf.concat([K.mean(ToDown[:, :2], axis=1, keepdims=True), ToDown[:, 2:]], axis=1)
    return K.mean(tf.signal.frame(ToDown[:,:,0], 2,2, axis=1), axis=-1)   


if __name__ == "__main__":
    
    BatchSize = 800
    
    TrainInp = np.load('../ProcessedData/ABPInp_Train.npy',allow_pickle=True)
    TrainOut = np.load('../ProcessedData/HypoOut_Train.npy',allow_pickle=True)
    ValInp = np.load('../ProcessedData/ABPInp_Val.npy',allow_pickle=True)
    ValOut = np.load('../ProcessedData/HypoOut_Val.npy',allow_pickle=True)
    
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) 
    with strategy.scope():

        Shapelet1Size = 30
        PaddSideForSim = 12
        FrameSize = 50
        AttSize = 5

#         b = 0.08 
#         N = int(np.ceil((4 / b)))
#         if not N % 2:  
#             N += 1  
#         RanVar = tf.constant(np.arange(N), dtype=tf.float32)
#         PaddSideForConv = (N-1)//2


        InputVec = Input(shape=(9000), name='Input')


        ### DWT 1st level
        FC1 = Conv1D(filters= 1,kernel_size= 20, strides=20//5, activation='softplus')(InputVec[:,:,None])
        FC1 = MaxPooling1D(pool_size=20,strides=20//5)(FC1)
        FC1 = Conv1D(filters= 1,kernel_size= 10, strides=10//2)(FC1)
        FC1 = MaxPooling1D(pool_size=10,strides=10//2)(FC1)[:,:,0]
        FC1 = Dense(10, activation='relu')(FC1)
        FC1 = Dense(1, activation='sigmoid')(FC1)
        FC1 = FC1*(0.5-K.epsilon())+K.epsilon()
        LP1, HP1 =  FilterGen (FC1)

        InputVecPad = tf.signal.frame(InputVec, N, 1)
        LP1_res = K.sum(InputVecPad * LP1[:,None,:], axis=-1, keepdims=True)
        HP1_res = K.sum(InputVecPad * HP1[:,None,:], axis=-1, keepdims=True)

        ## DownSampling
        LP1_Down = DownSampling(LP1_res)


        ### DWT 2nd level
        FC2 = Conv1D(filters= 1,kernel_size= 20, strides=20//5, activation='softplus')(LP1_Down[:,:,None])
        FC2 = MaxPooling1D(pool_size=20,strides=20//5)(FC2)
        FC2 = Conv1D(filters= 1,kernel_size= 10, strides=10//2)(FC2)
        FC2 = MaxPooling1D(pool_size=10,strides=10//2)(FC2)[:,:,0]
        FC2 = Dense(10, activation='relu')(FC2)
        FC2 = Dense(1, activation='sigmoid')(FC2)
        FC2 = FC2*(0.5-K.epsilon())+K.epsilon()
        LP2, HP2 =  FilterGen (FC2)

        LP1_DownPad = tf.signal.frame(LP1_Down, N, 1)
        LP2_res = K.sum(LP1_DownPad * LP2[:,None,:], axis=-1, keepdims=True)
        HP2_res = K.sum(LP1_DownPad * HP2[:,None,:], axis=-1, keepdims=True)

        ## DownSampling
        LP2_Down = DownSampling(LP2_res)


        ### DWT 3rd level
        FC3 = Conv1D(filters= 1,kernel_size= 10, strides=10//4, activation='softplus')(LP2_Down[:,:,None])
        FC3 = MaxPooling1D(pool_size=10,strides=10//4)(FC3)
        FC3 = Conv1D(filters= 1,kernel_size= 10, strides=10//2)(FC3)
        FC3 = MaxPooling1D(pool_size=10,strides=10//2)(FC3)[:,:,0]
        FC3 = Dense(10, activation='relu')(FC3)
        FC3 = Dense(1, activation='sigmoid')(FC3)
        FC3 = FC3*(0.5-K.epsilon())+K.epsilon()
        LP3, HP3 =  FilterGen (FC3)

        LP2_DownPad = tf.signal.frame(LP2_Down, N, 1)
        LP3_res = K.sum(LP2_DownPad * LP3[:,None,:], axis=-1, keepdims=True)
        HP3_res = K.sum(LP2_DownPad * HP3[:,None,:], axis=-1, keepdims=True)

        ## DownSampling
        LP3_Down = DownSampling(LP3_res)


        ### DWT 4th level
        FC4 = Conv1D(filters= 1,kernel_size= 10, strides=10//2, activation='softplus')(LP3_Down[:,:,None])
        FC4 = MaxPooling1D(pool_size=10,strides=10//2)(FC4)
        FC4 = Conv1D(filters= 1,kernel_size= 5, strides=5//2)(FC4)
        FC4 = MaxPooling1D(pool_size=5,strides=5//2)(FC4)[:,:,0]
        FC4 = Dense(10, activation='relu')(FC4)
        FC4 = Dense(1, activation='sigmoid')(FC4)
        FC4 = FC4*(0.5-K.epsilon())+K.epsilon()
        LP4, HP4 =  FilterGen (FC4)

        LP3_DownPad = tf.signal.frame(LP3_Down, N, 1)
        LP4_res = K.sum(LP3_DownPad * LP4[:,None,:], axis=-1, keepdims=True)

        ## DownSampling
        LP4_Down = DownSampling(LP4_res)

        
        ### DWT 5th level
        FC5 = Conv1D(filters= 1,kernel_size= 10, strides=10//2, activation='softplus')(LP4_Down[:,:,None])
        FC5 = MaxPooling1D(pool_size=10,strides=10//2)(FC5)
        FC5 = Conv1D(filters= 1,kernel_size= 5, strides=5//2)(FC5)
        FC5 = MaxPooling1D(pool_size=5,strides=5//2)(FC5)[:,:,0]
        FC5 = Dense(10, activation='relu')(FC5)
        FC5 = Dense(1, activation='sigmoid')(FC5)
        FC5 = FC5*(0.5-K.epsilon())+K.epsilon()
        LP5, HP5 =  FilterGen (FC5)

        LP4_DownPad = tf.signal.frame(LP4_Down, N, 1)
        LP5_res = K.sum(LP4_DownPad * LP5[:,None,:], axis=-1, keepdims=True)

        ## DownSampling
        LP5_Down = DownSampling(LP5_res)
        LP5_FeatDim = tf.signal.frame(LP5_Down, FrameSize, 1)
        
        ## Interval Wise Att and select interval
        LP5_ATT = Conv1D(filters= 1,kernel_size= FrameSize, activation='softplus', strides=1,padding="same")(Dropout(0.0)(LP5_Down[:,:,None]))
        LP5_ATT = MaxPooling1D(pool_size=FrameSize//2,strides=FrameSize//4)(LP5_ATT)
        LP5_ATT = Dense(30, activation='relu')(LP5_ATT[:,:,0])
        LP5_Mus = Dense(AttSize, activation='sigmoid')(LP5_ATT)


        dist = tfp.distributions.Normal( LP5_Mus[:,:,None], 0.075, name='Normal') 
        RandVec = tf.constant(np.linspace(0, 1, LP5_FeatDim.shape[1]), dtype=tf.float32)[None,None]
        RandVec = tf.tile(RandVec, (K.shape(LP5_Mus)[0], AttSize, 1))
        KerVal = dist.prob(RandVec)
        MinKV = K.min(KerVal, axis=-1, keepdims=True)
        MaxKV = K.max(KerVal, axis=-1, keepdims=True)
        KvRate = (KerVal - MinKV)/(MaxKV - MinKV)
        AggKvRate = K.sum(KvRate, axis=1)


        ### Gen Shape
        GenVecLayer = DoGenVec([Shapelet1Size, FrameSize])
        GenVec = Activation('sigmoid')(GenVecLayer(InputVec)) 

        LP5_X_sqare = K.sum(K.square(LP5_FeatDim), axis=2,keepdims=True)
        LP5_Y_sqare = K.sum(K.square(GenVec[:]), axis=1)[None,None]
        LP5_XY = tf.matmul(LP5_FeatDim, GenVec[:], transpose_b=True)
        LP5_Dist = (LP5_X_sqare + LP5_Y_sqare - 2*LP5_XY) 
        LP5_Dist_sqrt = K.sqrt(LP5_Dist+K.epsilon())

        ### Cluster Loss
        LP5_Dist_exp = tf.tile(LP5_Dist[:,None], (1,AttSize,1,1)) 
        KvRate_MaxInd = K.argmax(KvRate, axis=-1) 
        Att_Loss = K.min(tf.gather_nd(LP5_Dist_exp,KvRate_MaxInd[:,:,None], batch_dims=2 ), axis=-1)
        ClLoss = K.mean(Att_Loss * K.max(KvRate, axis=-1))


        ### Features
        Features_W = K.exp(-LP5_Dist_sqrt) * AggKvRate[:,:,None]
        Features = K.max(Features_W, axis=1) 
        BinOut = Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=0),name='BinOut')(Dropout(0.0)(Features))

        BinModel = Model(InputVec, BinOut)
        BinModel.add_loss(ClLoss)
        BinModel.add_metric(ClLoss, name='ClLoss')

        lrate = 0.0005
        decay = 1e-6
        adam = tf.keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=decay)
        BinModel.compile(loss='binary_crossentropy', optimizer='adam',  metrics={'BinOut':['binary_crossentropy','AUC']}, ) 


        # Run
        SaveFilePath = './Logs/SaveModel_{epoch:d}_{binary_crossentropy:.5f}_{auc:.5f}_{ClLoss:.5f}_{val_binary_crossentropy:.5f}_{val_auc:.5f}_{val_ClLoss:.5f}.hdf5'
        checkpoint = ModelCheckpoint(SaveFilePath,monitor=('val_loss'),verbose=0, save_best_only=True, mode='auto' ,period=1) 
        earlystopper = EarlyStopping(monitor='val_loss', patience=600, verbose=1,restore_best_weights=True)
        history = LossHistory()

        BinModel.fit(TrainInp, TrainOut, validation_data = (ValInp,ValOut), verbose=1, epochs=5000, callbacks=[history,earlystopper,checkpoint], )
        SaveFilePath = './Logs/ShapeBin_end.hdf5'
        BinModel.save(SaveFilePath)  