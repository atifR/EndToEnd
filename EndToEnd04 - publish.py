# -*- coding: utf-8 -*-
"""

@author: Atif 
"""


import tensorflow as tf
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import squareform
import sys
import time

FCNetModelPath = 'savedFCNetModel/model.ckpt'
ETEModelPath = 'savedModelETE03/model.ckpt'
summaryPath = 'logETE03/summaryLog'
NNModelPath = 'savedModelNN-NYU/model.ckpt'
resultsPath = 'resutlsETE'
datasetName = 'NYU'
nEpochs = 50
batch_size = 20


tf.reset_default_graph()        
init = tf.truncated_normal_initializer(stddev=0.1)

config = tf.ConfigProto()   # config for tensorflow to use GPU memory dynamically
config.gpu_options.allow_growth = True

def loadMatlabVar(matlabFileName, varName):
    mat_contents = sio.loadmat(matlabFileName)
    
    return mat_contents[varName]

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    healthyTag = 0
    patientTag = 1
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==patientTag:
           TP += 1
    for i in range(len(y_hat)): 
        if y_hat[i]==patientTag and y_actual[i] !=y_hat[i]:
           FP += 1
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==healthyTag:
           TN += 1
    for i in range(len(y_hat)): 
        if y_hat[i]==healthyTag and y_actual[i] !=y_hat[i]:
           FN += 1
    return(TP, FP, TN, FN)

def getTrainableParamNumber():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
        #    print(dim)
            variable_parameters *= dim.value
        #print(variable_parameters)
        print("variabel {} params {}".format(variable.name,variable_parameters))
        total_parameters += variable_parameters
    return total_parameters

def getSpecSens(gt,predictions):
    #gt = [x-1 for x in gt]
    #gt =  np.array(gt,dtype='int')
    #' gt is now in 0 and 1 values
    
    TP, FP, TN, FN = perf_measure(gt,predictions)
    sens = TP/(TP + FN)
    spec = TN/(TN + FP)
    return spec,sens

def extractLabelFromPheno(pheno):
    colId = 5
    labels = pheno[:,colId]
    newLabel = [1 if x!=0 else int(x) for x in labels]
    newLabel = [0 if x==0 else x for x in newLabel]
    newLabel =  np.array(newLabel,dtype='int')
    return newLabel

def Conv1D(x, nChannels, nameParam = 'conv', kernelSize = 3):
    with tf.name_scope('Conv'):
        # x is in format  [batch, length, channels]
        # create weight var format [inChannel,outChannel, filters]
        inChannels = x.get_shape()[2]
        w = tf.get_variable(name=nameParam+'_w',shape=[kernelSize,inChannels,nChannels],dtype = tf.float32, initializer=init)
        b = tf.get_variable(name=nameParam+'_b',dtype = tf.float32, initializer=tf.constant(0.01, shape=[nChannels], dtype=tf.float32))
        return tf.nn.bias_add(tf.nn.conv1d(value = x,filters=w, stride=1,padding = 'VALID', name = nameParam),b)

def batchNorm(x,nameParam='BN', inChannels=0):
    # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    with tf.name_scope(nameParam):
        #inChannels = x.get_shape()[2]
        batch_mean, batch_var = tf.nn.moments(x,[0])
        #scale = tf.Variable(tf.ones([inChannels]))
        #offset = tf.Variable(tf.zeros([inChannels]))
        
        offset  = tf.get_variable(name=nameParam+'_offset',shape=[inChannels],dtype = tf.float32, initializer=tf.zeros_initializer() )
        scale  = tf.get_variable(name=nameParam+'_scale',shape=[inChannels],dtype = tf.float32, initializer= tf.ones_initializer() )
        return tf.nn.batch_normalization(x,batch_mean, batch_var,offset,scale,0.01,name=nameParam)

def batchNormWithWeights(x,offset,scale,nameParam='BN'):
    with tf.name_scope(nameParam):
        #inChannels = x.get_shape()[2]
        batch_mean, batch_var = tf.nn.moments(x,[0])
        return tf.nn.batch_normalization(x,batch_mean,batch_var,offset,scale,0.01,name=nameParam)

def Conv1DWithWeights(x, w, b, nChannels, nameParam = 'conv', kernelSize = 3):
    with tf.name_scope(nameParam):
        # x is in format  [batch, length, channels]    
        return tf.nn.bias_add(tf.nn.conv1d(value = x,filters=w, stride=1,padding = 'VALID', name = nameParam),b)
    
def LeakyReLU(x, nameParam='LeakyReLU'):
    with tf.name_scope(nameParam):
        alpha = tf.get_variable(name=nameParam+'_w',shape=[1],dtype = tf.float32, initializer=init)
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def LeakyReLUWithWeights(x, alpha, nameParam='LeakyReLU'):
    with tf.name_scope(nameParam):
        #alpha = tf.get_variable(name=nameParam+'_w',shape=[1],dtype = tf.float32, initializer=init)
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def FullyConnected(x,nodes,nameParam="fc"):
    with tf.name_scope(nameParam):
        inNodes = x.get_shape()[1]
        w = tf.get_variable(name=nameParam+'_w',shape=[inNodes,nodes],dtype = tf.float32, initializer=init)
        b = tf.get_variable(name=nameParam+'_b',dtype = tf.float32, initializer=tf.constant(0.01, shape=[nodes], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(x,w),b,name = nameParam)
        return fc

def FullyConnectedWithWeights(x,w,b,nodes,nameParam="fc"):
    with tf.name_scope(nameParam):
        fc = tf.nn.bias_add(tf.matmul(x,w),b,name = nameParam)
        return fc

# Feature extractor network 
def makeRow(x):
        with tf.name_scope('FS_row'):
            #x = tf.placeholder(tf.float32, shape=[None, 172])
            #x = tf.reshape(x, [-1, 172, 1])
            x1 = Conv1D(x,32,'Conv1')
            #bn1 = tf.layers.batch_normalization(x1,name='BN1')
            bn1 = batchNorm(x1,'BN1')
            #activ1 = tf.nn.relu(bn1)
            activ1 = LeakyReLU(bn1,'LReLU1')
            pool1 = tf.layers.max_pooling1d(activ1,pool_size=2,strides=2, name = 'maxPool1')    
            
            c2 = Conv1D(pool1,64,'Conv2')
            bn2 = batchNorm(c2,'BN2')
            #activ2 = tf.nn.relu(bn2)
            activ2 = LeakyReLU(bn2,'LReLU2')
            pool2 = tf.layers.max_pooling1d(activ2,pool_size=2,strides=2, name = 'maxPool1')    
            
            c3 = Conv1D(pool2,96,'Conv3')
            bn3 = batchNorm(c3,'BN3')
            #activ3 = tf.nn.relu(bn3)
            activ3 = LeakyReLU(bn3,'LReLU3')
            
            c4 = Conv1D(activ3,64,'Conv4')
            c5 = Conv1D(c4,64,'Conv5')
            
            pool3 = tf.layers.max_pooling1d(c5,pool_size=2,strides=2, name = 'maxPool3')    
            
            #flattenSize = pool3.shape[1]*pool3.shape[2]
            flatten = tf.reshape(pool3,[-1,17*64],name='flatten')
            # dense 32        
            fc = FullyConnected(flatten,32,'FC')
            return fc

# Similarity measure network             
def makeDiffNN(tensor1,tensor2):
    with tf.name_scope('DiffNN'):
        merged = tf.concat([tensor1,tensor2],1,'merge')        
        fc1 = FullyConnected(merged,32,'FC1')
        fc2 = FullyConnected(fc1,32,'FC2')
        predictions = FullyConnected(fc2,2,'predictions')
        return predictions

'''
the function makes a row (feature extractor network) of FCNet and loads the saved wieghts and all variables
'''
def loadRow(x,weights):
#    with tf.Session() as sess:
#        saver = tf.train.import_meta_graph(modelPath + '.meta')
#        saver.restore(sess,modelPath)
        
#        graph = tf.get_default_graph()  #to get default graph
    with tf.name_scope('FS_row'):
        
        #wVal = w.eval()
        #bVal = b.eval()
        wTensor = tf.get_variable('Conv1_w',initializer=tf.constant(weights['Conv1_w']))
        bTensor = tf.get_variable('Conv1_b',initializer=tf.constant(weights['Conv1_b']))
        x1 = Conv1DWithWeights(x, wTensor,bTensor,32,'Conv1')
        #bn1 = tf.layers.batch_normal ization(x1,name='BN1')
        
        bn1Offset = tf.get_variable('BN1_offset',initializer=tf.constant(weights['BN1_offset']))
        bn1Scale = tf.get_variable('BN1_scale',initializer=tf.constant(weights['BN1_scale']))
        #bn1 = batchNormWithWeights(x1,weights['BN1_offset'],weights['BN1_scale'],'BN1')
        bn1 = batchNormWithWeights(x1,bn1Offset,bn1Scale,'BN1')
        #activ1 = tf.nn.relu(bn1)
        
        alphaTensor = tf.get_variable('LReLU1_w',initializer=tf.constant(weights['LReLU1_w']))
        activ1 = LeakyReLUWithWeights(bn1,alphaTensor,'LReLU1')
        pool1 = tf.layers.max_pooling1d(activ1,pool_size=2,strides=2, name = 'maxPool1')    
        
        
        wTensor = tf.get_variable('Conv2_w',initializer=tf.constant(weights['Conv2_w']))
        bTensor = tf.get_variable('Conv2_b',initializer=tf.constant(weights['Conv2_b']))
        
        c2 = Conv1DWithWeights(pool1,wTensor,bTensor,64,'Conv2')
        
        bn2Offset = tf.get_variable('BN2_offset',initializer=tf.constant(weights['BN2_offset']))
        bn2Scale = tf.get_variable('BN2_scale',initializer=tf.constant(weights['BN2_scale']))
        
        #bn2 = batchNormWithWeights(c2,weights['BN2_offset'],weights['BN2_scale'],'BN2')
        bn2 = batchNormWithWeights(c2,bn2Offset,bn2Scale,'BN2')
        #activ2 = tf.nn.relu(bn2)
        alphaTensor = tf.get_variable('LReLU2_w',initializer=tf.constant(weights['LReLU2_w']))
        activ2 = LeakyReLUWithWeights(bn2,alphaTensor,'LReLU2')
        pool2 = tf.layers.max_pooling1d(activ2,pool_size=2,strides=2, name = 'maxPool1')    
        
        wTensor = tf.get_variable('Conv3_w',initializer=tf.constant(weights['Conv3_w']))
        bTensor = tf.get_variable('Conv3_b',initializer=tf.constant(weights['Conv3_b']))
        c3 = Conv1DWithWeights(pool2,wTensor,bTensor,96,'Conv3')
        
        bn3Offset = tf.get_variable('BN3_offset',initializer=tf.constant(weights['BN3_offset']))
        bn3Scale = tf.get_variable('BN3_scale',initializer=tf.constant(weights['BN3_scale']))
        
        #bn3 = batchNormWithWeights(c3,weights['BN3_offset'],weights['BN3_scale'],'BN3')
        bn3 = batchNormWithWeights(c3,bn3Offset,bn3Scale,'BN3')
        #activ3 = tf.nn.relu(bn3)
        
        alphaTensor = tf.get_variable('LReLU3_w',initializer=tf.constant(weights['LReLU3_w']))
        activ3 = LeakyReLUWithWeights(bn3,alphaTensor,'LReLU3')
        
        
        wTensor = tf.get_variable('Conv4_w',initializer=tf.constant(weights['Conv4_w']))
        bTensor = tf.get_variable('Conv4_b',initializer=tf.constant(weights['Conv4_b']))
        c4 = Conv1DWithWeights(activ3,wTensor,bTensor,64,'Conv4')
        
        
        wTensor = tf.get_variable('Conv5_w',initializer=tf.constant(weights['Conv5_w']))
        bTensor = tf.get_variable('Conv5_b',initializer=tf.constant(weights['Conv5_b']))
        c5 = Conv1DWithWeights(c4,wTensor,bTensor,64,'Conv5')
        
        pool3 = tf.layers.max_pooling1d(c5,pool_size=2,strides=2, name = 'maxPool3')    
        
        #flattenSize = pool3.shape[1]*pool3.shape[2]
        flatten = tf.reshape(pool3,[-1,17*64],name='flatten')
        # dense 32        
        
        wTensor = tf.get_variable('FC_w',initializer=tf.constant(weights['FC_w']))
        bTensor = tf.get_variable('FC_b',initializer=tf.constant(weights['FC_b']))
        fc = FullyConnectedWithWeights(flatten,wTensor,bTensor,32,'FC')
        return fc

# load a saved feature extractor network 
def loadDiffNN(tensor1,tensor2,modelPath):
    with tf.name_scope('DiffNN'):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(modelPath + '.meta')
            saver.restore(sess,modelPath)
            graph = tf.get_default_graph()  #to get default graph
            
            merged = tf.concat([tensor1,tensor2],1,'merge')        
            
            w = graph.get_tensor_by_name('FC1_w:0').eval()
            b = graph.get_tensor_by_name('FC1_b:0').eval()
            wTensor = tf.get_variable('FC1_w',initializer=tf.constant(w))
            bTensor = tf.get_variable('FC1_b',initializer=tf.constant(b))
            fc1 = FullyConnectedWithWeights(merged,wTensor,bTensor,32,'FC1')
            
            w = graph.get_tensor_by_name('FC2_w:0').eval()
            b = graph.get_tensor_by_name('FC2_b:0').eval()
            wTensor = tf.get_variable('FC2_w',initializer=tf.constant(w))
            bTensor = tf.get_variable('FC2_b',initializer=tf.constant(b))
            fc2 = FullyConnectedWithWeights(fc1,wTensor,bTensor,32,'FC2')
            
            w = graph.get_tensor_by_name('predictions_w:0').eval()
            b = graph.get_tensor_by_name('predictions_b:0').eval()
            wTensor = tf.get_variable('predictions_w',initializer=tf.constant(w))
            bTensor = tf.get_variable('predictions_b',initializer=tf.constant(b))
            predictions = FullyConnectedWithWeights(fc2,wTensor,bTensor,2,'predictions')
        return predictions

    
## 2nd part of the model that is classification netwrok  


def makeNN(x):
    with tf.name_scope('NN'):
        layer1 = FullyConnected(x,100,'NN/FC1')
        bn1 = layer1
        bn1 = batchNorm(layer1,'NN/BN1',100)
        activ1 = bn1
        #activ1 = tf.nn.relu(bn1,'relu1') # LeakyReLU(bn1,'NN/relu1') 
        #activ1 = tf.nn.dropout(activ1,0.1)
        #layer2 = activ1
        layer2 = FullyConnected(activ1,50,'NN/FC2')
        bn2 = layer2
        bn2 = batchNorm(layer2,'NN/BN2',50)
        activ2 = bn2
        #activ2 = tf.nn.relu(bn2) # LeakyReLU(bn2,'NN/relu2')  
        
        layer3 = activ2
        layer3 = FullyConnected(activ2,50,'NN/FC3')
        #layer3 = batchNorm(layer3,'NN/BN3',50)
        
        #activ2 = tf.nn.dropout(activ2,0.1)
        predictions = FullyConnected(layer3,2,'NN/predictions')
    return predictions

# load saved parameter from disk to an array 
def getParamFromSavedNN(modelPath):
    weights = {}
    tempGraph = tf.Graph()      # get rid of defaul graph. donot put every thing there 
    with tf.Session(graph = tempGraph) as sess: 
        saver = tf.train.import_meta_graph(modelPath + '.meta')
        saver.restore(sess,modelPath)       
        graph = tf.get_default_graph()  #to get default graph
        print('Loading the weights ....')
        weights['NN/FC1_w'] = graph.get_tensor_by_name('NN/FC1_w:0').eval()
        weights['NN/FC1_b'] = graph.get_tensor_by_name('NN/FC1_b:0').eval()
        
        weights['NN/BN1_offset'] = graph.get_tensor_by_name('NN/BN1_offset:0').eval()
        weights['NN/BN1_scale'] = graph.get_tensor_by_name('NN/BN1_scale:0').eval()
        
        #weights['NN/relu1_w'] = graph.get_tensor_by_name('NN/relu1_w:0').eval()
        
        weights['NN/FC2_w'] = graph.get_tensor_by_name('NN/FC2_w:0').eval()
        weights['NN/FC2_b'] = graph.get_tensor_by_name('NN/FC2_b:0').eval()
        
        weights['NN/BN2_offset'] = graph.get_tensor_by_name('NN/BN2_offset:0').eval()
        weights['NN/BN2_scale'] = graph.get_tensor_by_name('NN/BN2_scale:0').eval()
        
        #weights['NN/relu2_w'] = graph.get_tensor_by_name('NN/relu2_w:0').eval()
        
        weights['NN/FC3_w'] = graph.get_tensor_by_name('NN/FC3_w:0').eval()
        weights['NN/FC3_b'] = graph.get_tensor_by_name('NN/FC3_b:0').eval()
                
        weights['NN/predictions_w'] = graph.get_tensor_by_name('NN/predictions_w:0').eval()
        weights['NN/predictions_b'] = graph.get_tensor_by_name('NN/predictions_b:0').eval()
        
        
        
    #tf.reset_default_graph()
    return weights
# load similarity measure network 
def loadNN(x,weights):
    with tf.name_scope('NN'):
        wTensor = tf.get_variable('NN/FC1_w',initializer=tf.constant(weights['NN/FC1_w']))
        bTensor = tf.get_variable('NN/FC1_b',initializer=tf.constant(weights['NN/FC1_b']))
        layer1 = FullyConnectedWithWeights(x,wTensor,bTensor,100,'NN/FC1')
        
        bn1 = layer1
        bn1Offset = tf.get_variable('NN/BN1_offset',initializer=tf.constant(weights['NN/BN1_offset']))
        bn1Scale = tf.get_variable('NN/BN1_scale',initializer=tf.constant(weights['NN/BN1_scale']))
        bn1 = batchNormWithWeights(layer1,bn1Offset,bn1Scale,'NN/BN1')
        activ1 = bn1
        #activ1 = tf.nn.relu(bn1)
        #relu1W = tf.get_variable('NN/relu1_w',initializer=tf.constant(weights['NN/relu1_w']))
        #activ1 = LeakyReLUWithWeights(activ1, relu1W, 'NN/relu1')
        #activ1 = LeakyReLU(activ1, 'NN/relu1')
        #activ1 = tf.nn.dropout(activ1,0.6)
        
        w2Tensor = tf.get_variable('NN/FC2_w',initializer=tf.constant(weights['NN/FC2_w']))
        b2Tensor = tf.get_variable('NN/FC2_b',initializer=tf.constant(weights['NN/FC2_b']))
        layer2 = FullyConnectedWithWeights(activ1, w2Tensor,b2Tensor,50,'NN/FC2')
        bn2 = layer2
        
        bn2Offset = tf.get_variable('NN/BN2_offset',initializer=tf.constant(weights['NN/BN2_offset']))
        bn2Scale = tf.get_variable('NN/BN2_scale',initializer=tf.constant(weights['NN/BN2_scale']))
        bn2 = batchNormWithWeights(layer2,bn2Offset,bn2Scale,'NN/BN2')
        #bn2 = tf.layers.batch_normalization(layer2,name = 'BN2')
        activ2 = bn2
        #activ2 = tf.nn.relu(bn2)
        #relu2W = tf.get_variable('NN/relu2_w',initializer=tf.constant(weights['NN/relu2_w']))
        #activ2 = LeakyReLUWithWeights(bn2, relu2W, 'NN/relu2')
        #activ2 = LeakyReLU(bn2,'NN/relu2')
        #activ2 = tf.nn.dropout(activ2,0.6)
        
        w3Tensor = tf.get_variable('NN/FC3_w',initializer=tf.constant(weights['NN/FC3_w']))
        b3Tensor = tf.get_variable('NN/FC3_b',initializer=tf.constant(weights['NN/FC3_b']))
        layer3 = FullyConnectedWithWeights(activ2, w3Tensor,b3Tensor,50,'NN/FC3')
        
        w4Tensor = tf.get_variable('NN/predictions_w',initializer=tf.constant(weights['NN/predictions_w']))
        b4Tensor = tf.get_variable('NN/predictions_b',initializer=tf.constant(weights['NN/predictions_b']))
        predictions = FullyConnectedWithWeights(layer3, w4Tensor, b4Tensor, 2,'predictions')
    return predictions
        






# main routine to run the training 
def runMain():
    # load the saved mdel and save the weights from the model 
    weights = {}
    with tf.Session() as sess: 
        saver = tf.train.import_meta_graph(FCNetModelPath + '.meta')
        saver.restore(sess,FCNetModelPath)       
        graph = tf.get_default_graph()  #to get default graph
        print('Loading the weights ....')
        diffNN_w1 = graph.get_tensor_by_name('DiffNNet/FC1_w:0').eval()
        diffNN_b1 = graph.get_tensor_by_name('DiffNNet/FC1_b:0').eval()
        diffNN_w2 = graph.get_tensor_by_name('DiffNNet/FC2_w:0').eval()
        diffNN_b2 = graph.get_tensor_by_name('DiffNNet/FC2_b:0').eval()
        diffNN_w3 = graph.get_tensor_by_name('DiffNNet/predictions_w:0').eval()
        diffNN_b3 = graph.get_tensor_by_name('DiffNNet/predictions_b:0').eval()
        
        weights['Conv1_w'] = graph.get_tensor_by_name('siameseNet/Conv1_w:0').eval()
        weights['Conv1_b'] = graph.get_tensor_by_name('siameseNet/Conv1_b:0').eval()
        
        weights['BN1_offset'] = graph.get_tensor_by_name('siameseNet/BN1_offset:0').eval()
        weights['BN1_scale'] = graph.get_tensor_by_name('siameseNet/BN1_scale:0').eval()
        
        weights['LReLU1_w'] = graph.get_tensor_by_name('siameseNet/LReLU1_w:0').eval()
        
        weights['Conv2_w'] = graph.get_tensor_by_name('siameseNet/Conv2_w:0').eval()
        weights['Conv2_b'] = graph.get_tensor_by_name('siameseNet/Conv2_b:0').eval()
        
        weights['BN2_offset'] = graph.get_tensor_by_name('siameseNet/BN2_offset:0').eval()
        weights['BN2_scale'] = graph.get_tensor_by_name('siameseNet/BN2_scale:0').eval()
        
        weights['LReLU2_w'] = graph.get_tensor_by_name('siameseNet/LReLU2_w:0').eval()
        
        weights['Conv3_w'] = graph.get_tensor_by_name('siameseNet/Conv3_w:0').eval()
        weights['Conv3_b'] = graph.get_tensor_by_name('siameseNet/Conv3_b:0').eval()
        
        weights['BN3_offset'] = graph.get_tensor_by_name('siameseNet/BN3_offset:0').eval()
        weights['BN3_scale'] = graph.get_tensor_by_name('siameseNet/BN3_scale:0').eval()
        
        weights['LReLU3_w'] = graph.get_tensor_by_name('siameseNet/LReLU3_w:0').eval()
        
        weights['Conv4_w'] = graph.get_tensor_by_name('siameseNet/Conv4_w:0').eval()
        weights['Conv4_b'] = graph.get_tensor_by_name('siameseNet/Conv4_b:0').eval()
        
        weights['Conv5_w'] = graph.get_tensor_by_name('siameseNet/Conv5_w:0').eval()
        weights['Conv5_b'] = graph.get_tensor_by_name('siameseNet/Conv5_b:0').eval()
        
        weights['FC_w'] = graph.get_tensor_by_name('siameseNet/FC_w:0').eval()
        weights['FC_b'] = graph.get_tensor_by_name('siameseNet/FC_b:0').eval()
    
    
    tf.reset_default_graph()            
    x = tf.placeholder(tf.float32, shape=[None, 90,172,1],name='input_x')
    #x1 = tf.placeholder(tf.float32, shape=[None, 172,1])
    y = tf.placeholder(tf.float32, shape=[None, 2],name='input_y')
    
    nRegs = x.shape[1]
    combinations = 4005 # nRegs * (nRegs-1)/2
    combinations_temp = 4005
    
    print('creating tensors for feature nets and diff NNets')
    p = [tf.Tensor for _ in range(nRegs)]
    fc = [tf.Tensor for _ in range(combinations_temp)]
    print('creating lookup tables')
    a = [p for p in range(0,combinations)]
    a = np.array(a,dtype="int")
    lookupMat = squareform(a)
    
    
    
    with tf.Session() as sess:
        #saver = tf.train.import_meta_graph(FCNetModelPath + '.meta')
        #saver.restore(sess,FCNetModelPath)       
        #graph = tf.get_default_graph()  #to get default graph
        print('Loading the feature nets with weights ....')
        with tf.name_scope('FeatureNets_ns'):
            with tf.variable_scope("siameseNet_vs") as scope:
                x1 = tf.slice(x,[0,0,0,0],[-1,1,-1,-1])
                x1 = tf.reshape(x1,[-1,172,1])
                p[0] = loadRow(x1,weights)
                #p[0] = loadRow(x[:,0,:,:],weights)
                #p[0] = makeRow(x[:,0,:,:])
                scope.reuse_variables()
                for i in range(1,nRegs):
                    #p[i] = loadRow(x[:,i,:,:],weights)
                    p[i] = loadRow(  tf.reshape(tf.slice(x,[0,i,0,0],[-1,1,-1,-1]),[-1,172,1]) , weights)
                    #  loadRow(x[:,i,:,:] , weights)
                    #p[i] = makeRow(x[:,i,:,:])
                    print('{}-'.format(i),end='')
        
        print('Loading the Diff nets with weights ....')        
        # sorry for some hard-coding here. it was all to make the prog run faster .. :P 
        with tf.name_scope('DiffNet_ns'):
            with tf.variable_scope("DiffNNet") as scope:
                # retirieving tensor values
                            
                merged = tf.concat([p[0],p[1]],1,'merge')        
                wTensor = tf.get_variable('FC1_w',initializer=tf.constant(diffNN_w1))
                bTensor = tf.get_variable('FC1_b',initializer=tf.constant(diffNN_b1))
                fc1 = FullyConnectedWithWeights(merged,wTensor,bTensor,32,'FC1')
                
                wTensor = tf.get_variable('FC2_w',initializer=tf.constant(diffNN_w2))
                bTensor = tf.get_variable('FC2_b',initializer=tf.constant(diffNN_b2))
                fc2 = FullyConnectedWithWeights(fc1,wTensor,bTensor,32,'FC2')
                
                wTensor = tf.get_variable('predictions_w',initializer=tf.constant(diffNN_w3))
                bTensor = tf.get_variable('predictions_b',initializer=tf.constant(diffNN_b3))
                fc[0] = FullyConnectedWithWeights(fc2,wTensor,bTensor,2,'predictions')
                fc[0] = tf.nn.softmax(fc[0])
    #            fc[0] = makeDiffNN(p[0],p[1])
                scope.reuse_variables()
                for i in range(combinations_temp):
                    #a = datetime.datetime.now()
                    r,c = np.where(lookupMat==i)
                    
                    merged = tf.concat([ p[r[0]], p[c[0]] ],1,'merge')        
                    wTensor = tf.get_variable('FC1_w',initializer=tf.constant(diffNN_w1))
                    bTensor = tf.get_variable('FC1_b',initializer=tf.constant(diffNN_b1))
                    fc1 = FullyConnectedWithWeights(merged,wTensor,bTensor,32,'FC1')
                    
                    wTensor = tf.get_variable('FC2_w',initializer=tf.constant(diffNN_w2))
                    bTensor = tf.get_variable('FC2_b',initializer=tf.constant(diffNN_b2))
                    fc2 = FullyConnectedWithWeights(fc1,wTensor,bTensor,32,'FC2')
                    
                    wTensor = tf.get_variable('predictions_w',initializer=tf.constant(diffNN_w3))
                    bTensor = tf.get_variable('predictions_b',initializer=tf.constant(diffNN_b3))
                    fc[i] = FullyConnectedWithWeights(fc2,wTensor,bTensor,2,'predictions')
                    
                    fc[i] = tf.nn.softmax(fc[i])
    
    #                fc[i] = makeDiffNN(p[r[0]], p[c[0]])
                    print('{}-'.format(i),end='')
                    if ( i % 50 ==0):
                        sys.stdout.flush()
    print('Creating NN part of model...')
    with tf.name_scope('mergedSc'):
        #mergedFC = tf.concat([p for p in fc],1,'mergedFC')
        mergedFC = tf.concat([fc[0],fc[1],fc[2],fc[3],fc[4],fc[5],fc[6],fc[7],fc[8],fc[9],fc[10],fc[11],fc[12],fc[13],fc[14],fc[15],fc[16],fc[17],fc[18],fc[19],fc[20],fc[21],fc[22],fc[23],fc[24],fc[25],fc[26],fc[27],fc[28],fc[29],fc[30],fc[31],fc[32],fc[33],fc[34],fc[35],fc[36],fc[37],fc[38],fc[39],fc[40],fc[41],fc[42],fc[43],fc[44],fc[45],fc[46],fc[47],fc[48],fc[49],fc[50],fc[51],fc[52],fc[53],fc[54],fc[55],fc[56],fc[57],fc[58],fc[59],fc[60],fc[61],fc[62],fc[63],fc[64],fc[65],fc[66],fc[67],fc[68],fc[69],fc[70],fc[71],fc[72],fc[73],fc[74],fc[75],fc[76],fc[77],fc[78],fc[79],fc[80],fc[81],fc[82],fc[83],fc[84],fc[85],fc[86],fc[87],fc[88],fc[89],fc[90],fc[91],fc[92],fc[93],fc[94],fc[95],fc[96],fc[97],fc[98],fc[99],fc[100],fc[101],fc[102],fc[103],fc[104],fc[105],fc[106],fc[107],fc[108],fc[109],fc[110],fc[111],fc[112],fc[113],fc[114],fc[115],fc[116],fc[117],fc[118],fc[119],fc[120],fc[121],fc[122],fc[123],fc[124],fc[125],fc[126],fc[127],fc[128],fc[129],fc[130],fc[131],fc[132],fc[133],fc[134],fc[135],fc[136],fc[137],fc[138],fc[139],fc[140],fc[141],fc[142],fc[143],fc[144],fc[145],fc[146],fc[147],fc[148],fc[149],fc[150],fc[151],fc[152],fc[153],fc[154],fc[155],fc[156],fc[157],fc[158],fc[159],fc[160],fc[161],fc[162],fc[163],fc[164],fc[165],fc[166],fc[167],fc[168],fc[169],fc[170],fc[171],fc[172],fc[173],fc[174],fc[175],fc[176],fc[177],fc[178],fc[179],fc[180],fc[181],fc[182],fc[183],fc[184],fc[185],fc[186],fc[187],fc[188],fc[189],fc[190],fc[191],fc[192],fc[193],fc[194],fc[195],fc[196],fc[197],fc[198],fc[199],fc[200],fc[201],fc[202],fc[203],fc[204],fc[205],fc[206],fc[207],fc[208],fc[209],fc[210],fc[211],fc[212],fc[213],fc[214],fc[215],fc[216],fc[217],fc[218],fc[219],fc[220],fc[221],fc[222],fc[223],fc[224],fc[225],fc[226],fc[227],fc[228],fc[229],fc[230],fc[231],fc[232],fc[233],fc[234],fc[235],fc[236],fc[237],fc[238],fc[239],fc[240],fc[241],fc[242],fc[243],fc[244],fc[245],fc[246],fc[247],fc[248],fc[249],fc[250],fc[251],fc[252],fc[253],fc[254],fc[255],fc[256],fc[257],fc[258],fc[259],fc[260],fc[261],fc[262],fc[263],fc[264],fc[265],fc[266],fc[267],fc[268],fc[269],fc[270],fc[271],fc[272],fc[273],fc[274],fc[275],fc[276],fc[277],fc[278],fc[279],fc[280],fc[281],fc[282],fc[283],fc[284],fc[285],fc[286],fc[287],fc[288],fc[289],fc[290],fc[291],fc[292],fc[293],fc[294],fc[295],fc[296],fc[297],fc[298],fc[299],fc[300],fc[301],fc[302],fc[303],fc[304],fc[305],fc[306],fc[307],fc[308],fc[309],fc[310],fc[311],fc[312],fc[313],fc[314],fc[315],fc[316],fc[317],fc[318],fc[319],fc[320],fc[321],fc[322],fc[323],fc[324],fc[325],fc[326],fc[327],fc[328],fc[329],fc[330],fc[331],fc[332],fc[333],fc[334],fc[335],fc[336],fc[337],fc[338],fc[339],fc[340],fc[341],fc[342],fc[343],fc[344],fc[345],fc[346],fc[347],fc[348],fc[349],fc[350],fc[351],fc[352],fc[353],fc[354],fc[355],fc[356],fc[357],fc[358],fc[359],fc[360],fc[361],fc[362],fc[363],fc[364],fc[365],fc[366],fc[367],fc[368],fc[369],fc[370],fc[371],fc[372],fc[373],fc[374],fc[375],fc[376],fc[377],fc[378],fc[379],fc[380],fc[381],fc[382],fc[383],fc[384],fc[385],fc[386],fc[387],fc[388],fc[389],fc[390],fc[391],fc[392],fc[393],fc[394],fc[395],fc[396],fc[397],fc[398],fc[399],fc[400],fc[401],fc[402],fc[403],fc[404],fc[405],fc[406],fc[407],fc[408],fc[409],fc[410],fc[411],fc[412],fc[413],fc[414],fc[415],fc[416],fc[417],fc[418],fc[419],fc[420],fc[421],fc[422],fc[423],fc[424],fc[425],fc[426],fc[427],fc[428],fc[429],fc[430],fc[431],fc[432],fc[433],fc[434],fc[435],fc[436],fc[437],fc[438],fc[439],fc[440],fc[441],fc[442],fc[443],fc[444],fc[445],fc[446],fc[447],fc[448],fc[449],fc[450],fc[451],fc[452],fc[453],fc[454],fc[455],fc[456],fc[457],fc[458],fc[459],fc[460],fc[461],fc[462],fc[463],fc[464],fc[465],fc[466],fc[467],fc[468],fc[469],fc[470],fc[471],fc[472],fc[473],fc[474],fc[475],fc[476],fc[477],fc[478],fc[479],fc[480],fc[481],fc[482],fc[483],fc[484],fc[485],fc[486],fc[487],fc[488],fc[489],fc[490],fc[491],fc[492],fc[493],fc[494],fc[495],fc[496],fc[497],fc[498],fc[499],fc[500],fc[501],fc[502],fc[503],fc[504],fc[505],fc[506],fc[507],fc[508],fc[509],fc[510],fc[511],fc[512],fc[513],fc[514],fc[515],fc[516],fc[517],fc[518],fc[519],fc[520],fc[521],fc[522],fc[523],fc[524],fc[525],fc[526],fc[527],fc[528],fc[529],fc[530],fc[531],fc[532],fc[533],fc[534],fc[535],fc[536],fc[537],fc[538],fc[539],fc[540],fc[541],fc[542],fc[543],fc[544],fc[545],fc[546],fc[547],fc[548],fc[549],fc[550],fc[551],fc[552],fc[553],fc[554],fc[555],fc[556],fc[557],fc[558],fc[559],fc[560],fc[561],fc[562],fc[563],fc[564],fc[565],fc[566],fc[567],fc[568],fc[569],fc[570],fc[571],fc[572],fc[573],fc[574],fc[575],fc[576],fc[577],fc[578],fc[579],fc[580],fc[581],fc[582],fc[583],fc[584],fc[585],fc[586],fc[587],fc[588],fc[589],fc[590],fc[591],fc[592],fc[593],fc[594],fc[595],fc[596],fc[597],fc[598],fc[599],fc[600],fc[601],fc[602],fc[603],fc[604],fc[605],fc[606],fc[607],fc[608],fc[609],fc[610],fc[611],fc[612],fc[613],fc[614],fc[615],fc[616],fc[617],fc[618],fc[619],fc[620],fc[621],fc[622],fc[623],fc[624],fc[625],fc[626],fc[627],fc[628],fc[629],fc[630],fc[631],fc[632],fc[633],fc[634],fc[635],fc[636],fc[637],fc[638],fc[639],fc[640],fc[641],fc[642],fc[643],fc[644],fc[645],fc[646],fc[647],fc[648],fc[649],fc[650],fc[651],fc[652],fc[653],fc[654],fc[655],fc[656],fc[657],fc[658],fc[659],fc[660],fc[661],fc[662],fc[663],fc[664],fc[665],fc[666],fc[667],fc[668],fc[669],fc[670],fc[671],fc[672],fc[673],fc[674],fc[675],fc[676],fc[677],fc[678],fc[679],fc[680],fc[681],fc[682],fc[683],fc[684],fc[685],fc[686],fc[687],fc[688],fc[689],fc[690],fc[691],fc[692],fc[693],fc[694],fc[695],fc[696],fc[697],fc[698],fc[699],fc[700],fc[701],fc[702],fc[703],fc[704],fc[705],fc[706],fc[707],fc[708],fc[709],fc[710],fc[711],fc[712],fc[713],fc[714],fc[715],fc[716],fc[717],fc[718],fc[719],fc[720],fc[721],fc[722],fc[723],fc[724],fc[725],fc[726],fc[727],fc[728],fc[729],fc[730],fc[731],fc[732],fc[733],fc[734],fc[735],fc[736],fc[737],fc[738],fc[739],fc[740],fc[741],fc[742],fc[743],fc[744],fc[745],fc[746],fc[747],fc[748],fc[749],fc[750],fc[751],fc[752],fc[753],fc[754],fc[755],fc[756],fc[757],fc[758],fc[759],fc[760],fc[761],fc[762],fc[763],fc[764],fc[765],fc[766],fc[767],fc[768],fc[769],fc[770],fc[771],fc[772],fc[773],fc[774],fc[775],fc[776],fc[777],fc[778],fc[779],fc[780],fc[781],fc[782],fc[783],fc[784],fc[785],fc[786],fc[787],fc[788],fc[789],fc[790],fc[791],fc[792],fc[793],fc[794],fc[795],fc[796],fc[797],fc[798],fc[799],fc[800],fc[801],fc[802],fc[803],fc[804],fc[805],fc[806],fc[807],fc[808],fc[809],fc[810],fc[811],fc[812],fc[813],fc[814],fc[815],fc[816],fc[817],fc[818],fc[819],fc[820],fc[821],fc[822],fc[823],fc[824],fc[825],fc[826],fc[827],fc[828],fc[829],fc[830],fc[831],fc[832],fc[833],fc[834],fc[835],fc[836],fc[837],fc[838],fc[839],fc[840],fc[841],fc[842],fc[843],fc[844],fc[845],fc[846],fc[847],fc[848],fc[849],fc[850],fc[851],fc[852],fc[853],fc[854],fc[855],fc[856],fc[857],fc[858],fc[859],fc[860],fc[861],fc[862],fc[863],fc[864],fc[865],fc[866],fc[867],fc[868],fc[869],fc[870],fc[871],fc[872],fc[873],fc[874],fc[875],fc[876],fc[877],fc[878],fc[879],fc[880],fc[881],fc[882],fc[883],fc[884],fc[885],fc[886],fc[887],fc[888],fc[889],fc[890],fc[891],fc[892],fc[893],fc[894],fc[895],fc[896],fc[897],fc[898],fc[899],fc[900],fc[901],fc[902],fc[903],fc[904],fc[905],fc[906],fc[907],fc[908],fc[909],fc[910],fc[911],fc[912],fc[913],fc[914],fc[915],fc[916],fc[917],fc[918],fc[919],fc[920],fc[921],fc[922],fc[923],fc[924],fc[925],fc[926],fc[927],fc[928],fc[929],fc[930],fc[931],fc[932],fc[933],fc[934],fc[935],fc[936],fc[937],fc[938],fc[939],fc[940],fc[941],fc[942],fc[943],fc[944],fc[945],fc[946],fc[947],fc[948],fc[949],fc[950],fc[951],fc[952],fc[953],fc[954],fc[955],fc[956],fc[957],fc[958],fc[959],fc[960],fc[961],fc[962],fc[963],fc[964],fc[965],fc[966],fc[967],fc[968],fc[969],fc[970],fc[971],fc[972],fc[973],fc[974],fc[975],fc[976],fc[977],fc[978],fc[979],fc[980],fc[981],fc[982],fc[983],fc[984],fc[985],fc[986],fc[987],fc[988],fc[989],fc[990],fc[991],fc[992],fc[993],fc[994],fc[995],fc[996],fc[997],fc[998],fc[999],fc[1000],fc[1001],fc[1002],fc[1003],fc[1004],fc[1005],fc[1006],fc[1007],fc[1008],fc[1009],fc[1010],fc[1011],fc[1012],fc[1013],fc[1014],fc[1015],fc[1016],fc[1017],fc[1018],fc[1019],fc[1020],fc[1021],fc[1022],fc[1023],fc[1024],fc[1025],fc[1026],fc[1027],fc[1028],fc[1029],fc[1030],fc[1031],fc[1032],fc[1033],fc[1034],fc[1035],fc[1036],fc[1037],fc[1038],fc[1039],fc[1040],fc[1041],fc[1042],fc[1043],fc[1044],fc[1045],fc[1046],fc[1047],fc[1048],fc[1049],fc[1050],fc[1051],fc[1052],fc[1053],fc[1054],fc[1055],fc[1056],fc[1057],fc[1058],fc[1059],fc[1060],fc[1061],fc[1062],fc[1063],fc[1064],fc[1065],fc[1066],fc[1067],fc[1068],fc[1069],fc[1070],fc[1071],fc[1072],fc[1073],fc[1074],fc[1075],fc[1076],fc[1077],fc[1078],fc[1079],fc[1080],fc[1081],fc[1082],fc[1083],fc[1084],fc[1085],fc[1086],fc[1087],fc[1088],fc[1089],fc[1090],fc[1091],fc[1092],fc[1093],fc[1094],fc[1095],fc[1096],fc[1097],fc[1098],fc[1099],fc[1100],fc[1101],fc[1102],fc[1103],fc[1104],fc[1105],fc[1106],fc[1107],fc[1108],fc[1109],fc[1110],fc[1111],fc[1112],fc[1113],fc[1114],fc[1115],fc[1116],fc[1117],fc[1118],fc[1119],fc[1120],fc[1121],fc[1122],fc[1123],fc[1124],fc[1125],fc[1126],fc[1127],fc[1128],fc[1129],fc[1130],fc[1131],fc[1132],fc[1133],fc[1134],fc[1135],fc[1136],fc[1137],fc[1138],fc[1139],fc[1140],fc[1141],fc[1142],fc[1143],fc[1144],fc[1145],fc[1146],fc[1147],fc[1148],fc[1149],fc[1150],fc[1151],fc[1152],fc[1153],fc[1154],fc[1155],fc[1156],fc[1157],fc[1158],fc[1159],fc[1160],fc[1161],fc[1162],fc[1163],fc[1164],fc[1165],fc[1166],fc[1167],fc[1168],fc[1169],fc[1170],fc[1171],fc[1172],fc[1173],fc[1174],fc[1175],fc[1176],fc[1177],fc[1178],fc[1179],fc[1180],fc[1181],fc[1182],fc[1183],fc[1184],fc[1185],fc[1186],fc[1187],fc[1188],fc[1189],fc[1190],fc[1191],fc[1192],fc[1193],fc[1194],fc[1195],fc[1196],fc[1197],fc[1198],fc[1199],fc[1200],fc[1201],fc[1202],fc[1203],fc[1204],fc[1205],fc[1206],fc[1207],fc[1208],fc[1209],fc[1210],fc[1211],fc[1212],fc[1213],fc[1214],fc[1215],fc[1216],fc[1217],fc[1218],fc[1219],fc[1220],fc[1221],fc[1222],fc[1223],fc[1224],fc[1225],fc[1226],fc[1227],fc[1228],fc[1229],fc[1230],fc[1231],fc[1232],fc[1233],fc[1234],fc[1235],fc[1236],fc[1237],fc[1238],fc[1239],fc[1240],fc[1241],fc[1242],fc[1243],fc[1244],fc[1245],fc[1246],fc[1247],fc[1248],fc[1249],fc[1250],fc[1251],fc[1252],fc[1253],fc[1254],fc[1255],fc[1256],fc[1257],fc[1258],fc[1259],fc[1260],fc[1261],fc[1262],fc[1263],fc[1264],fc[1265],fc[1266],fc[1267],fc[1268],fc[1269],fc[1270],fc[1271],fc[1272],fc[1273],fc[1274],fc[1275],fc[1276],fc[1277],fc[1278],fc[1279],fc[1280],fc[1281],fc[1282],fc[1283],fc[1284],fc[1285],fc[1286],fc[1287],fc[1288],fc[1289],fc[1290],fc[1291],fc[1292],fc[1293],fc[1294],fc[1295],fc[1296],fc[1297],fc[1298],fc[1299],fc[1300],fc[1301],fc[1302],fc[1303],fc[1304],fc[1305],fc[1306],fc[1307],fc[1308],fc[1309],fc[1310],fc[1311],fc[1312],fc[1313],fc[1314],fc[1315],fc[1316],fc[1317],fc[1318],fc[1319],fc[1320],fc[1321],fc[1322],fc[1323],fc[1324],fc[1325],fc[1326],fc[1327],fc[1328],fc[1329],fc[1330],fc[1331],fc[1332],fc[1333],fc[1334],fc[1335],fc[1336],fc[1337],fc[1338],fc[1339],fc[1340],fc[1341],fc[1342],fc[1343],fc[1344],fc[1345],fc[1346],fc[1347],fc[1348],fc[1349],fc[1350],fc[1351],fc[1352],fc[1353],fc[1354],fc[1355],fc[1356],fc[1357],fc[1358],fc[1359],fc[1360],fc[1361],fc[1362],fc[1363],fc[1364],fc[1365],fc[1366],fc[1367],fc[1368],fc[1369],fc[1370],fc[1371],fc[1372],fc[1373],fc[1374],fc[1375],fc[1376],fc[1377],fc[1378],fc[1379],fc[1380],fc[1381],fc[1382],fc[1383],fc[1384],fc[1385],fc[1386],fc[1387],fc[1388],fc[1389],fc[1390],fc[1391],fc[1392],fc[1393],fc[1394],fc[1395],fc[1396],fc[1397],fc[1398],fc[1399],fc[1400],fc[1401],fc[1402],fc[1403],fc[1404],fc[1405],fc[1406],fc[1407],fc[1408],fc[1409],fc[1410],fc[1411],fc[1412],fc[1413],fc[1414],fc[1415],fc[1416],fc[1417],fc[1418],fc[1419],fc[1420],fc[1421],fc[1422],fc[1423],fc[1424],fc[1425],fc[1426],fc[1427],fc[1428],fc[1429],fc[1430],fc[1431],fc[1432],fc[1433],fc[1434],fc[1435],fc[1436],fc[1437],fc[1438],fc[1439],fc[1440],fc[1441],fc[1442],fc[1443],fc[1444],fc[1445],fc[1446],fc[1447],fc[1448],fc[1449],fc[1450],fc[1451],fc[1452],fc[1453],fc[1454],fc[1455],fc[1456],fc[1457],fc[1458],fc[1459],fc[1460],fc[1461],fc[1462],fc[1463],fc[1464],fc[1465],fc[1466],fc[1467],fc[1468],fc[1469],fc[1470],fc[1471],fc[1472],fc[1473],fc[1474],fc[1475],fc[1476],fc[1477],fc[1478],fc[1479],fc[1480],fc[1481],fc[1482],fc[1483],fc[1484],fc[1485],fc[1486],fc[1487],fc[1488],fc[1489],fc[1490],fc[1491],fc[1492],fc[1493],fc[1494],fc[1495],fc[1496],fc[1497],fc[1498],fc[1499],fc[1500],fc[1501],fc[1502],fc[1503],fc[1504],fc[1505],fc[1506],fc[1507],fc[1508],fc[1509],fc[1510],fc[1511],fc[1512],fc[1513],fc[1514],fc[1515],fc[1516],fc[1517],fc[1518],fc[1519],fc[1520],fc[1521],fc[1522],fc[1523],fc[1524],fc[1525],fc[1526],fc[1527],fc[1528],fc[1529],fc[1530],fc[1531],fc[1532],fc[1533],fc[1534],fc[1535],fc[1536],fc[1537],fc[1538],fc[1539],fc[1540],fc[1541],fc[1542],fc[1543],fc[1544],fc[1545],fc[1546],fc[1547],fc[1548],fc[1549],fc[1550],fc[1551],fc[1552],fc[1553],fc[1554],fc[1555],fc[1556],fc[1557],fc[1558],fc[1559],fc[1560],fc[1561],fc[1562],fc[1563],fc[1564],fc[1565],fc[1566],fc[1567],fc[1568],fc[1569],fc[1570],fc[1571],fc[1572],fc[1573],fc[1574],fc[1575],fc[1576],fc[1577],fc[1578],fc[1579],fc[1580],fc[1581],fc[1582],fc[1583],fc[1584],fc[1585],fc[1586],fc[1587],fc[1588],fc[1589],fc[1590],fc[1591],fc[1592],fc[1593],fc[1594],fc[1595],fc[1596],fc[1597],fc[1598],fc[1599],fc[1600],fc[1601],fc[1602],fc[1603],fc[1604],fc[1605],fc[1606],fc[1607],fc[1608],fc[1609],fc[1610],fc[1611],fc[1612],fc[1613],fc[1614],fc[1615],fc[1616],fc[1617],fc[1618],fc[1619],fc[1620],fc[1621],fc[1622],fc[1623],fc[1624],fc[1625],fc[1626],fc[1627],fc[1628],fc[1629],fc[1630],fc[1631],fc[1632],fc[1633],fc[1634],fc[1635],fc[1636],fc[1637],fc[1638],fc[1639],fc[1640],fc[1641],fc[1642],fc[1643],fc[1644],fc[1645],fc[1646],fc[1647],fc[1648],fc[1649],fc[1650],fc[1651],fc[1652],fc[1653],fc[1654],fc[1655],fc[1656],fc[1657],fc[1658],fc[1659],fc[1660],fc[1661],fc[1662],fc[1663],fc[1664],fc[1665],fc[1666],fc[1667],fc[1668],fc[1669],fc[1670],fc[1671],fc[1672],fc[1673],fc[1674],fc[1675],fc[1676],fc[1677],fc[1678],fc[1679],fc[1680],fc[1681],fc[1682],fc[1683],fc[1684],fc[1685],fc[1686],fc[1687],fc[1688],fc[1689],fc[1690],fc[1691],fc[1692],fc[1693],fc[1694],fc[1695],fc[1696],fc[1697],fc[1698],fc[1699],fc[1700],fc[1701],fc[1702],fc[1703],fc[1704],fc[1705],fc[1706],fc[1707],fc[1708],fc[1709],fc[1710],fc[1711],fc[1712],fc[1713],fc[1714],fc[1715],fc[1716],fc[1717],fc[1718],fc[1719],fc[1720],fc[1721],fc[1722],fc[1723],fc[1724],fc[1725],fc[1726],fc[1727],fc[1728],fc[1729],fc[1730],fc[1731],fc[1732],fc[1733],fc[1734],fc[1735],fc[1736],fc[1737],fc[1738],fc[1739],fc[1740],fc[1741],fc[1742],fc[1743],fc[1744],fc[1745],fc[1746],fc[1747],fc[1748],fc[1749],fc[1750],fc[1751],fc[1752],fc[1753],fc[1754],fc[1755],fc[1756],fc[1757],fc[1758],fc[1759],fc[1760],fc[1761],fc[1762],fc[1763],fc[1764],fc[1765],fc[1766],fc[1767],fc[1768],fc[1769],fc[1770],fc[1771],fc[1772],fc[1773],fc[1774],fc[1775],fc[1776],fc[1777],fc[1778],fc[1779],fc[1780],fc[1781],fc[1782],fc[1783],fc[1784],fc[1785],fc[1786],fc[1787],fc[1788],fc[1789],fc[1790],fc[1791],fc[1792],fc[1793],fc[1794],fc[1795],fc[1796],fc[1797],fc[1798],fc[1799],fc[1800],fc[1801],fc[1802],fc[1803],fc[1804],fc[1805],fc[1806],fc[1807],fc[1808],fc[1809],fc[1810],fc[1811],fc[1812],fc[1813],fc[1814],fc[1815],fc[1816],fc[1817],fc[1818],fc[1819],fc[1820],fc[1821],fc[1822],fc[1823],fc[1824],fc[1825],fc[1826],fc[1827],fc[1828],fc[1829],fc[1830],fc[1831],fc[1832],fc[1833],fc[1834],fc[1835],fc[1836],fc[1837],fc[1838],fc[1839],fc[1840],fc[1841],fc[1842],fc[1843],fc[1844],fc[1845],fc[1846],fc[1847],fc[1848],fc[1849],fc[1850],fc[1851],fc[1852],fc[1853],fc[1854],fc[1855],fc[1856],fc[1857],fc[1858],fc[1859],fc[1860],fc[1861],fc[1862],fc[1863],fc[1864],fc[1865],fc[1866],fc[1867],fc[1868],fc[1869],fc[1870],fc[1871],fc[1872],fc[1873],fc[1874],fc[1875],fc[1876],fc[1877],fc[1878],fc[1879],fc[1880],fc[1881],fc[1882],fc[1883],fc[1884],fc[1885],fc[1886],fc[1887],fc[1888],fc[1889],fc[1890],fc[1891],fc[1892],fc[1893],fc[1894],fc[1895],fc[1896],fc[1897],fc[1898],fc[1899],fc[1900],fc[1901],fc[1902],fc[1903],fc[1904],fc[1905],fc[1906],fc[1907],fc[1908],fc[1909],fc[1910],fc[1911],fc[1912],fc[1913],fc[1914],fc[1915],fc[1916],fc[1917],fc[1918],fc[1919],fc[1920],fc[1921],fc[1922],fc[1923],fc[1924],fc[1925],fc[1926],fc[1927],fc[1928],fc[1929],fc[1930],fc[1931],fc[1932],fc[1933],fc[1934],fc[1935],fc[1936],fc[1937],fc[1938],fc[1939],fc[1940],fc[1941],fc[1942],fc[1943],fc[1944],fc[1945],fc[1946],fc[1947],fc[1948],fc[1949],fc[1950],fc[1951],fc[1952],fc[1953],fc[1954],fc[1955],fc[1956],fc[1957],fc[1958],fc[1959],fc[1960],fc[1961],fc[1962],fc[1963],fc[1964],fc[1965],fc[1966],fc[1967],fc[1968],fc[1969],fc[1970],fc[1971],fc[1972],fc[1973],fc[1974],fc[1975],fc[1976],fc[1977],fc[1978],fc[1979],fc[1980],fc[1981],fc[1982],fc[1983],fc[1984],fc[1985],fc[1986],fc[1987],fc[1988],fc[1989],fc[1990],fc[1991],fc[1992],fc[1993],fc[1994],fc[1995],fc[1996],fc[1997],fc[1998],fc[1999],fc[2000],fc[2001],fc[2002],fc[2003],fc[2004],fc[2005],fc[2006],fc[2007],fc[2008],fc[2009],fc[2010],fc[2011],fc[2012],fc[2013],fc[2014],fc[2015],fc[2016],fc[2017],fc[2018],fc[2019],fc[2020],fc[2021],fc[2022],fc[2023],fc[2024],fc[2025],fc[2026],fc[2027],fc[2028],fc[2029],fc[2030],fc[2031],fc[2032],fc[2033],fc[2034],fc[2035],fc[2036],fc[2037],fc[2038],fc[2039],fc[2040],fc[2041],fc[2042],fc[2043],fc[2044],fc[2045],fc[2046],fc[2047],fc[2048],fc[2049],fc[2050],fc[2051],fc[2052],fc[2053],fc[2054],fc[2055],fc[2056],fc[2057],fc[2058],fc[2059],fc[2060],fc[2061],fc[2062],fc[2063],fc[2064],fc[2065],fc[2066],fc[2067],fc[2068],fc[2069],fc[2070],fc[2071],fc[2072],fc[2073],fc[2074],fc[2075],fc[2076],fc[2077],fc[2078],fc[2079],fc[2080],fc[2081],fc[2082],fc[2083],fc[2084],fc[2085],fc[2086],fc[2087],fc[2088],fc[2089],fc[2090],fc[2091],fc[2092],fc[2093],fc[2094],fc[2095],fc[2096],fc[2097],fc[2098],fc[2099],fc[2100],fc[2101],fc[2102],fc[2103],fc[2104],fc[2105],fc[2106],fc[2107],fc[2108],fc[2109],fc[2110],fc[2111],fc[2112],fc[2113],fc[2114],fc[2115],fc[2116],fc[2117],fc[2118],fc[2119],fc[2120],fc[2121],fc[2122],fc[2123],fc[2124],fc[2125],fc[2126],fc[2127],fc[2128],fc[2129],fc[2130],fc[2131],fc[2132],fc[2133],fc[2134],fc[2135],fc[2136],fc[2137],fc[2138],fc[2139],fc[2140],fc[2141],fc[2142],fc[2143],fc[2144],fc[2145],fc[2146],fc[2147],fc[2148],fc[2149],fc[2150],fc[2151],fc[2152],fc[2153],fc[2154],fc[2155],fc[2156],fc[2157],fc[2158],fc[2159],fc[2160],fc[2161],fc[2162],fc[2163],fc[2164],fc[2165],fc[2166],fc[2167],fc[2168],fc[2169],fc[2170],fc[2171],fc[2172],fc[2173],fc[2174],fc[2175],fc[2176],fc[2177],fc[2178],fc[2179],fc[2180],fc[2181],fc[2182],fc[2183],fc[2184],fc[2185],fc[2186],fc[2187],fc[2188],fc[2189],fc[2190],fc[2191],fc[2192],fc[2193],fc[2194],fc[2195],fc[2196],fc[2197],fc[2198],fc[2199],fc[2200],fc[2201],fc[2202],fc[2203],fc[2204],fc[2205],fc[2206],fc[2207],fc[2208],fc[2209],fc[2210],fc[2211],fc[2212],fc[2213],fc[2214],fc[2215],fc[2216],fc[2217],fc[2218],fc[2219],fc[2220],fc[2221],fc[2222],fc[2223],fc[2224],fc[2225],fc[2226],fc[2227],fc[2228],fc[2229],fc[2230],fc[2231],fc[2232],fc[2233],fc[2234],fc[2235],fc[2236],fc[2237],fc[2238],fc[2239],fc[2240],fc[2241],fc[2242],fc[2243],fc[2244],fc[2245],fc[2246],fc[2247],fc[2248],fc[2249],fc[2250],fc[2251],fc[2252],fc[2253],fc[2254],fc[2255],fc[2256],fc[2257],fc[2258],fc[2259],fc[2260],fc[2261],fc[2262],fc[2263],fc[2264],fc[2265],fc[2266],fc[2267],fc[2268],fc[2269],fc[2270],fc[2271],fc[2272],fc[2273],fc[2274],fc[2275],fc[2276],fc[2277],fc[2278],fc[2279],fc[2280],fc[2281],fc[2282],fc[2283],fc[2284],fc[2285],fc[2286],fc[2287],fc[2288],fc[2289],fc[2290],fc[2291],fc[2292],fc[2293],fc[2294],fc[2295],fc[2296],fc[2297],fc[2298],fc[2299],fc[2300],fc[2301],fc[2302],fc[2303],fc[2304],fc[2305],fc[2306],fc[2307],fc[2308],fc[2309],fc[2310],fc[2311],fc[2312],fc[2313],fc[2314],fc[2315],fc[2316],fc[2317],fc[2318],fc[2319],fc[2320],fc[2321],fc[2322],fc[2323],fc[2324],fc[2325],fc[2326],fc[2327],fc[2328],fc[2329],fc[2330],fc[2331],fc[2332],fc[2333],fc[2334],fc[2335],fc[2336],fc[2337],fc[2338],fc[2339],fc[2340],fc[2341],fc[2342],fc[2343],fc[2344],fc[2345],fc[2346],fc[2347],fc[2348],fc[2349],fc[2350],fc[2351],fc[2352],fc[2353],fc[2354],fc[2355],fc[2356],fc[2357],fc[2358],fc[2359],fc[2360],fc[2361],fc[2362],fc[2363],fc[2364],fc[2365],fc[2366],fc[2367],fc[2368],fc[2369],fc[2370],fc[2371],fc[2372],fc[2373],fc[2374],fc[2375],fc[2376],fc[2377],fc[2378],fc[2379],fc[2380],fc[2381],fc[2382],fc[2383],fc[2384],fc[2385],fc[2386],fc[2387],fc[2388],fc[2389],fc[2390],fc[2391],fc[2392],fc[2393],fc[2394],fc[2395],fc[2396],fc[2397],fc[2398],fc[2399],fc[2400],fc[2401],fc[2402],fc[2403],fc[2404],fc[2405],fc[2406],fc[2407],fc[2408],fc[2409],fc[2410],fc[2411],fc[2412],fc[2413],fc[2414],fc[2415],fc[2416],fc[2417],fc[2418],fc[2419],fc[2420],fc[2421],fc[2422],fc[2423],fc[2424],fc[2425],fc[2426],fc[2427],fc[2428],fc[2429],fc[2430],fc[2431],fc[2432],fc[2433],fc[2434],fc[2435],fc[2436],fc[2437],fc[2438],fc[2439],fc[2440],fc[2441],fc[2442],fc[2443],fc[2444],fc[2445],fc[2446],fc[2447],fc[2448],fc[2449],fc[2450],fc[2451],fc[2452],fc[2453],fc[2454],fc[2455],fc[2456],fc[2457],fc[2458],fc[2459],fc[2460],fc[2461],fc[2462],fc[2463],fc[2464],fc[2465],fc[2466],fc[2467],fc[2468],fc[2469],fc[2470],fc[2471],fc[2472],fc[2473],fc[2474],fc[2475],fc[2476],fc[2477],fc[2478],fc[2479],fc[2480],fc[2481],fc[2482],fc[2483],fc[2484],fc[2485],fc[2486],fc[2487],fc[2488],fc[2489],fc[2490],fc[2491],fc[2492],fc[2493],fc[2494],fc[2495],fc[2496],fc[2497],fc[2498],fc[2499],fc[2500],fc[2501],fc[2502],fc[2503],fc[2504],fc[2505],fc[2506],fc[2507],fc[2508],fc[2509],fc[2510],fc[2511],fc[2512],fc[2513],fc[2514],fc[2515],fc[2516],fc[2517],fc[2518],fc[2519],fc[2520],fc[2521],fc[2522],fc[2523],fc[2524],fc[2525],fc[2526],fc[2527],fc[2528],fc[2529],fc[2530],fc[2531],fc[2532],fc[2533],fc[2534],fc[2535],fc[2536],fc[2537],fc[2538],fc[2539],fc[2540],fc[2541],fc[2542],fc[2543],fc[2544],fc[2545],fc[2546],fc[2547],fc[2548],fc[2549],fc[2550],fc[2551],fc[2552],fc[2553],fc[2554],fc[2555],fc[2556],fc[2557],fc[2558],fc[2559],fc[2560],fc[2561],fc[2562],fc[2563],fc[2564],fc[2565],fc[2566],fc[2567],fc[2568],fc[2569],fc[2570],fc[2571],fc[2572],fc[2573],fc[2574],fc[2575],fc[2576],fc[2577],fc[2578],fc[2579],fc[2580],fc[2581],fc[2582],fc[2583],fc[2584],fc[2585],fc[2586],fc[2587],fc[2588],fc[2589],fc[2590],fc[2591],fc[2592],fc[2593],fc[2594],fc[2595],fc[2596],fc[2597],fc[2598],fc[2599],fc[2600],fc[2601],fc[2602],fc[2603],fc[2604],fc[2605],fc[2606],fc[2607],fc[2608],fc[2609],fc[2610],fc[2611],fc[2612],fc[2613],fc[2614],fc[2615],fc[2616],fc[2617],fc[2618],fc[2619],fc[2620],fc[2621],fc[2622],fc[2623],fc[2624],fc[2625],fc[2626],fc[2627],fc[2628],fc[2629],fc[2630],fc[2631],fc[2632],fc[2633],fc[2634],fc[2635],fc[2636],fc[2637],fc[2638],fc[2639],fc[2640],fc[2641],fc[2642],fc[2643],fc[2644],fc[2645],fc[2646],fc[2647],fc[2648],fc[2649],fc[2650],fc[2651],fc[2652],fc[2653],fc[2654],fc[2655],fc[2656],fc[2657],fc[2658],fc[2659],fc[2660],fc[2661],fc[2662],fc[2663],fc[2664],fc[2665],fc[2666],fc[2667],fc[2668],fc[2669],fc[2670],fc[2671],fc[2672],fc[2673],fc[2674],fc[2675],fc[2676],fc[2677],fc[2678],fc[2679],fc[2680],fc[2681],fc[2682],fc[2683],fc[2684],fc[2685],fc[2686],fc[2687],fc[2688],fc[2689],fc[2690],fc[2691],fc[2692],fc[2693],fc[2694],fc[2695],fc[2696],fc[2697],fc[2698],fc[2699],fc[2700],fc[2701],fc[2702],fc[2703],fc[2704],fc[2705],fc[2706],fc[2707],fc[2708],fc[2709],fc[2710],fc[2711],fc[2712],fc[2713],fc[2714],fc[2715],fc[2716],fc[2717],fc[2718],fc[2719],fc[2720],fc[2721],fc[2722],fc[2723],fc[2724],fc[2725],fc[2726],fc[2727],fc[2728],fc[2729],fc[2730],fc[2731],fc[2732],fc[2733],fc[2734],fc[2735],fc[2736],fc[2737],fc[2738],fc[2739],fc[2740],fc[2741],fc[2742],fc[2743],fc[2744],fc[2745],fc[2746],fc[2747],fc[2748],fc[2749],fc[2750],fc[2751],fc[2752],fc[2753],fc[2754],fc[2755],fc[2756],fc[2757],fc[2758],fc[2759],fc[2760],fc[2761],fc[2762],fc[2763],fc[2764],fc[2765],fc[2766],fc[2767],fc[2768],fc[2769],fc[2770],fc[2771],fc[2772],fc[2773],fc[2774],fc[2775],fc[2776],fc[2777],fc[2778],fc[2779],fc[2780],fc[2781],fc[2782],fc[2783],fc[2784],fc[2785],fc[2786],fc[2787],fc[2788],fc[2789],fc[2790],fc[2791],fc[2792],fc[2793],fc[2794],fc[2795],fc[2796],fc[2797],fc[2798],fc[2799],fc[2800],fc[2801],fc[2802],fc[2803],fc[2804],fc[2805],fc[2806],fc[2807],fc[2808],fc[2809],fc[2810],fc[2811],fc[2812],fc[2813],fc[2814],fc[2815],fc[2816],fc[2817],fc[2818],fc[2819],fc[2820],fc[2821],fc[2822],fc[2823],fc[2824],fc[2825],fc[2826],fc[2827],fc[2828],fc[2829],fc[2830],fc[2831],fc[2832],fc[2833],fc[2834],fc[2835],fc[2836],fc[2837],fc[2838],fc[2839],fc[2840],fc[2841],fc[2842],fc[2843],fc[2844],fc[2845],fc[2846],fc[2847],fc[2848],fc[2849],fc[2850],fc[2851],fc[2852],fc[2853],fc[2854],fc[2855],fc[2856],fc[2857],fc[2858],fc[2859],fc[2860],fc[2861],fc[2862],fc[2863],fc[2864],fc[2865],fc[2866],fc[2867],fc[2868],fc[2869],fc[2870],fc[2871],fc[2872],fc[2873],fc[2874],fc[2875],fc[2876],fc[2877],fc[2878],fc[2879],fc[2880],fc[2881],fc[2882],fc[2883],fc[2884],fc[2885],fc[2886],fc[2887],fc[2888],fc[2889],fc[2890],fc[2891],fc[2892],fc[2893],fc[2894],fc[2895],fc[2896],fc[2897],fc[2898],fc[2899],fc[2900],fc[2901],fc[2902],fc[2903],fc[2904],fc[2905],fc[2906],fc[2907],fc[2908],fc[2909],fc[2910],fc[2911],fc[2912],fc[2913],fc[2914],fc[2915],fc[2916],fc[2917],fc[2918],fc[2919],fc[2920],fc[2921],fc[2922],fc[2923],fc[2924],fc[2925],fc[2926],fc[2927],fc[2928],fc[2929],fc[2930],fc[2931],fc[2932],fc[2933],fc[2934],fc[2935],fc[2936],fc[2937],fc[2938],fc[2939],fc[2940],fc[2941],fc[2942],fc[2943],fc[2944],fc[2945],fc[2946],fc[2947],fc[2948],fc[2949],fc[2950],fc[2951],fc[2952],fc[2953],fc[2954],fc[2955],fc[2956],fc[2957],fc[2958],fc[2959],fc[2960],fc[2961],fc[2962],fc[2963],fc[2964],fc[2965],fc[2966],fc[2967],fc[2968],fc[2969],fc[2970],fc[2971],fc[2972],fc[2973],fc[2974],fc[2975],fc[2976],fc[2977],fc[2978],fc[2979],fc[2980],fc[2981],fc[2982],fc[2983],fc[2984],fc[2985],fc[2986],fc[2987],fc[2988],fc[2989],fc[2990],fc[2991],fc[2992],fc[2993],fc[2994],fc[2995],fc[2996],fc[2997],fc[2998],fc[2999],fc[3000],fc[3001],fc[3002],fc[3003],fc[3004],fc[3005],fc[3006],fc[3007],fc[3008],fc[3009],fc[3010],fc[3011],fc[3012],fc[3013],fc[3014],fc[3015],fc[3016],fc[3017],fc[3018],fc[3019],fc[3020],fc[3021],fc[3022],fc[3023],fc[3024],fc[3025],fc[3026],fc[3027],fc[3028],fc[3029],fc[3030],fc[3031],fc[3032],fc[3033],fc[3034],fc[3035],fc[3036],fc[3037],fc[3038],fc[3039],fc[3040],fc[3041],fc[3042],fc[3043],fc[3044],fc[3045],fc[3046],fc[3047],fc[3048],fc[3049],fc[3050],fc[3051],fc[3052],fc[3053],fc[3054],fc[3055],fc[3056],fc[3057],fc[3058],fc[3059],fc[3060],fc[3061],fc[3062],fc[3063],fc[3064],fc[3065],fc[3066],fc[3067],fc[3068],fc[3069],fc[3070],fc[3071],fc[3072],fc[3073],fc[3074],fc[3075],fc[3076],fc[3077],fc[3078],fc[3079],fc[3080],fc[3081],fc[3082],fc[3083],fc[3084],fc[3085],fc[3086],fc[3087],fc[3088],fc[3089],fc[3090],fc[3091],fc[3092],fc[3093],fc[3094],fc[3095],fc[3096],fc[3097],fc[3098],fc[3099],fc[3100],fc[3101],fc[3102],fc[3103],fc[3104],fc[3105],fc[3106],fc[3107],fc[3108],fc[3109],fc[3110],fc[3111],fc[3112],fc[3113],fc[3114],fc[3115],fc[3116],fc[3117],fc[3118],fc[3119],fc[3120],fc[3121],fc[3122],fc[3123],fc[3124],fc[3125],fc[3126],fc[3127],fc[3128],fc[3129],fc[3130],fc[3131],fc[3132],fc[3133],fc[3134],fc[3135],fc[3136],fc[3137],fc[3138],fc[3139],fc[3140],fc[3141],fc[3142],fc[3143],fc[3144],fc[3145],fc[3146],fc[3147],fc[3148],fc[3149],fc[3150],fc[3151],fc[3152],fc[3153],fc[3154],fc[3155],fc[3156],fc[3157],fc[3158],fc[3159],fc[3160],fc[3161],fc[3162],fc[3163],fc[3164],fc[3165],fc[3166],fc[3167],fc[3168],fc[3169],fc[3170],fc[3171],fc[3172],fc[3173],fc[3174],fc[3175],fc[3176],fc[3177],fc[3178],fc[3179],fc[3180],fc[3181],fc[3182],fc[3183],fc[3184],fc[3185],fc[3186],fc[3187],fc[3188],fc[3189],fc[3190],fc[3191],fc[3192],fc[3193],fc[3194],fc[3195],fc[3196],fc[3197],fc[3198],fc[3199],fc[3200],fc[3201],fc[3202],fc[3203],fc[3204],fc[3205],fc[3206],fc[3207],fc[3208],fc[3209],fc[3210],fc[3211],fc[3212],fc[3213],fc[3214],fc[3215],fc[3216],fc[3217],fc[3218],fc[3219],fc[3220],fc[3221],fc[3222],fc[3223],fc[3224],fc[3225],fc[3226],fc[3227],fc[3228],fc[3229],fc[3230],fc[3231],fc[3232],fc[3233],fc[3234],fc[3235],fc[3236],fc[3237],fc[3238],fc[3239],fc[3240],fc[3241],fc[3242],fc[3243],fc[3244],fc[3245],fc[3246],fc[3247],fc[3248],fc[3249],fc[3250],fc[3251],fc[3252],fc[3253],fc[3254],fc[3255],fc[3256],fc[3257],fc[3258],fc[3259],fc[3260],fc[3261],fc[3262],fc[3263],fc[3264],fc[3265],fc[3266],fc[3267],fc[3268],fc[3269],fc[3270],fc[3271],fc[3272],fc[3273],fc[3274],fc[3275],fc[3276],fc[3277],fc[3278],fc[3279],fc[3280],fc[3281],fc[3282],fc[3283],fc[3284],fc[3285],fc[3286],fc[3287],fc[3288],fc[3289],fc[3290],fc[3291],fc[3292],fc[3293],fc[3294],fc[3295],fc[3296],fc[3297],fc[3298],fc[3299],fc[3300],fc[3301],fc[3302],fc[3303],fc[3304],fc[3305],fc[3306],fc[3307],fc[3308],fc[3309],fc[3310],fc[3311],fc[3312],fc[3313],fc[3314],fc[3315],fc[3316],fc[3317],fc[3318],fc[3319],fc[3320],fc[3321],fc[3322],fc[3323],fc[3324],fc[3325],fc[3326],fc[3327],fc[3328],fc[3329],fc[3330],fc[3331],fc[3332],fc[3333],fc[3334],fc[3335],fc[3336],fc[3337],fc[3338],fc[3339],fc[3340],fc[3341],fc[3342],fc[3343],fc[3344],fc[3345],fc[3346],fc[3347],fc[3348],fc[3349],fc[3350],fc[3351],fc[3352],fc[3353],fc[3354],fc[3355],fc[3356],fc[3357],fc[3358],fc[3359],fc[3360],fc[3361],fc[3362],fc[3363],fc[3364],fc[3365],fc[3366],fc[3367],fc[3368],fc[3369],fc[3370],fc[3371],fc[3372],fc[3373],fc[3374],fc[3375],fc[3376],fc[3377],fc[3378],fc[3379],fc[3380],fc[3381],fc[3382],fc[3383],fc[3384],fc[3385],fc[3386],fc[3387],fc[3388],fc[3389],fc[3390],fc[3391],fc[3392],fc[3393],fc[3394],fc[3395],fc[3396],fc[3397],fc[3398],fc[3399],fc[3400],fc[3401],fc[3402],fc[3403],fc[3404],fc[3405],fc[3406],fc[3407],fc[3408],fc[3409],fc[3410],fc[3411],fc[3412],fc[3413],fc[3414],fc[3415],fc[3416],fc[3417],fc[3418],fc[3419],fc[3420],fc[3421],fc[3422],fc[3423],fc[3424],fc[3425],fc[3426],fc[3427],fc[3428],fc[3429],fc[3430],fc[3431],fc[3432],fc[3433],fc[3434],fc[3435],fc[3436],fc[3437],fc[3438],fc[3439],fc[3440],fc[3441],fc[3442],fc[3443],fc[3444],fc[3445],fc[3446],fc[3447],fc[3448],fc[3449],fc[3450],fc[3451],fc[3452],fc[3453],fc[3454],fc[3455],fc[3456],fc[3457],fc[3458],fc[3459],fc[3460],fc[3461],fc[3462],fc[3463],fc[3464],fc[3465],fc[3466],fc[3467],fc[3468],fc[3469],fc[3470],fc[3471],fc[3472],fc[3473],fc[3474],fc[3475],fc[3476],fc[3477],fc[3478],fc[3479],fc[3480],fc[3481],fc[3482],fc[3483],fc[3484],fc[3485],fc[3486],fc[3487],fc[3488],fc[3489],fc[3490],fc[3491],fc[3492],fc[3493],fc[3494],fc[3495],fc[3496],fc[3497],fc[3498],fc[3499],fc[3500],fc[3501],fc[3502],fc[3503],fc[3504],fc[3505],fc[3506],fc[3507],fc[3508],fc[3509],fc[3510],fc[3511],fc[3512],fc[3513],fc[3514],fc[3515],fc[3516],fc[3517],fc[3518],fc[3519],fc[3520],fc[3521],fc[3522],fc[3523],fc[3524],fc[3525],fc[3526],fc[3527],fc[3528],fc[3529],fc[3530],fc[3531],fc[3532],fc[3533],fc[3534],fc[3535],fc[3536],fc[3537],fc[3538],fc[3539],fc[3540],fc[3541],fc[3542],fc[3543],fc[3544],fc[3545],fc[3546],fc[3547],fc[3548],fc[3549],fc[3550],fc[3551],fc[3552],fc[3553],fc[3554],fc[3555],fc[3556],fc[3557],fc[3558],fc[3559],fc[3560],fc[3561],fc[3562],fc[3563],fc[3564],fc[3565],fc[3566],fc[3567],fc[3568],fc[3569],fc[3570],fc[3571],fc[3572],fc[3573],fc[3574],fc[3575],fc[3576],fc[3577],fc[3578],fc[3579],fc[3580],fc[3581],fc[3582],fc[3583],fc[3584],fc[3585],fc[3586],fc[3587],fc[3588],fc[3589],fc[3590],fc[3591],fc[3592],fc[3593],fc[3594],fc[3595],fc[3596],fc[3597],fc[3598],fc[3599],fc[3600],fc[3601],fc[3602],fc[3603],fc[3604],fc[3605],fc[3606],fc[3607],fc[3608],fc[3609],fc[3610],fc[3611],fc[3612],fc[3613],fc[3614],fc[3615],fc[3616],fc[3617],fc[3618],fc[3619],fc[3620],fc[3621],fc[3622],fc[3623],fc[3624],fc[3625],fc[3626],fc[3627],fc[3628],fc[3629],fc[3630],fc[3631],fc[3632],fc[3633],fc[3634],fc[3635],fc[3636],fc[3637],fc[3638],fc[3639],fc[3640],fc[3641],fc[3642],fc[3643],fc[3644],fc[3645],fc[3646],fc[3647],fc[3648],fc[3649],fc[3650],fc[3651],fc[3652],fc[3653],fc[3654],fc[3655],fc[3656],fc[3657],fc[3658],fc[3659],fc[3660],fc[3661],fc[3662],fc[3663],fc[3664],fc[3665],fc[3666],fc[3667],fc[3668],fc[3669],fc[3670],fc[3671],fc[3672],fc[3673],fc[3674],fc[3675],fc[3676],fc[3677],fc[3678],fc[3679],fc[3680],fc[3681],fc[3682],fc[3683],fc[3684],fc[3685],fc[3686],fc[3687],fc[3688],fc[3689],fc[3690],fc[3691],fc[3692],fc[3693],fc[3694],fc[3695],fc[3696],fc[3697],fc[3698],fc[3699],fc[3700],fc[3701],fc[3702],fc[3703],fc[3704],fc[3705],fc[3706],fc[3707],fc[3708],fc[3709],fc[3710],fc[3711],fc[3712],fc[3713],fc[3714],fc[3715],fc[3716],fc[3717],fc[3718],fc[3719],fc[3720],fc[3721],fc[3722],fc[3723],fc[3724],fc[3725],fc[3726],fc[3727],fc[3728],fc[3729],fc[3730],fc[3731],fc[3732],fc[3733],fc[3734],fc[3735],fc[3736],fc[3737],fc[3738],fc[3739],fc[3740],fc[3741],fc[3742],fc[3743],fc[3744],fc[3745],fc[3746],fc[3747],fc[3748],fc[3749],fc[3750],fc[3751],fc[3752],fc[3753],fc[3754],fc[3755],fc[3756],fc[3757],fc[3758],fc[3759],fc[3760],fc[3761],fc[3762],fc[3763],fc[3764],fc[3765],fc[3766],fc[3767],fc[3768],fc[3769],fc[3770],fc[3771],fc[3772],fc[3773],fc[3774],fc[3775],fc[3776],fc[3777],fc[3778],fc[3779],fc[3780],fc[3781],fc[3782],fc[3783],fc[3784],fc[3785],fc[3786],fc[3787],fc[3788],fc[3789],fc[3790],fc[3791],fc[3792],fc[3793],fc[3794],fc[3795],fc[3796],fc[3797],fc[3798],fc[3799],fc[3800],fc[3801],fc[3802],fc[3803],fc[3804],fc[3805],fc[3806],fc[3807],fc[3808],fc[3809],fc[3810],fc[3811],fc[3812],fc[3813],fc[3814],fc[3815],fc[3816],fc[3817],fc[3818],fc[3819],fc[3820],fc[3821],fc[3822],fc[3823],fc[3824],fc[3825],fc[3826],fc[3827],fc[3828],fc[3829],fc[3830],fc[3831],fc[3832],fc[3833],fc[3834],fc[3835],fc[3836],fc[3837],fc[3838],fc[3839],fc[3840],fc[3841],fc[3842],fc[3843],fc[3844],fc[3845],fc[3846],fc[3847],fc[3848],fc[3849],fc[3850],fc[3851],fc[3852],fc[3853],fc[3854],fc[3855],fc[3856],fc[3857],fc[3858],fc[3859],fc[3860],fc[3861],fc[3862],fc[3863],fc[3864],fc[3865],fc[3866],fc[3867],fc[3868],fc[3869],fc[3870],fc[3871],fc[3872],fc[3873],fc[3874],fc[3875],fc[3876],fc[3877],fc[3878],fc[3879],fc[3880],fc[3881],fc[3882],fc[3883],fc[3884],fc[3885],fc[3886],fc[3887],fc[3888],fc[3889],fc[3890],fc[3891],fc[3892],fc[3893],fc[3894],fc[3895],fc[3896],fc[3897],fc[3898],fc[3899],fc[3900],fc[3901],fc[3902],fc[3903],fc[3904],fc[3905],fc[3906],fc[3907],fc[3908],fc[3909],fc[3910],fc[3911],fc[3912],fc[3913],fc[3914],fc[3915],fc[3916],fc[3917],fc[3918],fc[3919],fc[3920],fc[3921],fc[3922],fc[3923],fc[3924],fc[3925],fc[3926],fc[3927],fc[3928],fc[3929],fc[3930],fc[3931],fc[3932],fc[3933],fc[3934],fc[3935],fc[3936],fc[3937],fc[3938],fc[3939],fc[3940],fc[3941],fc[3942],fc[3943],fc[3944],fc[3945],fc[3946],fc[3947],fc[3948],fc[3949],fc[3950],fc[3951],fc[3952],fc[3953],fc[3954],fc[3955],fc[3956],fc[3957],fc[3958],fc[3959],fc[3960],fc[3961],fc[3962],fc[3963],fc[3964],fc[3965],fc[3966],fc[3967],fc[3968],fc[3969],fc[3970],fc[3971],fc[3972],fc[3973],fc[3974],fc[3975],fc[3976],fc[3977],fc[3978],fc[3979],fc[3980],fc[3981],fc[3982],fc[3983],fc[3984],fc[3985],fc[3986],fc[3987],fc[3988],fc[3989],fc[3990],fc[3991],fc[3992],fc[3993],fc[3994],fc[3995],fc[3996],fc[3997],fc[3998],fc[3999],fc[4000],fc[4001],fc[4002],fc[4003],fc[4004]],1,'mergedFC')
    evenVals = mergedFC[:,0::2]    #neglecting every second output of a netwrk, so retaining one out out i.e. FC value
    
    # build NN here 
    predictions = makeNN(evenVals)
    
    # or get NN here 
    #weights = getParamFromSavedNN(NNModelPath)
    #predictions = loadNN(evenVals,weights)
    
    ph_lr_nn = tf.placeholder(tf.float32,shape=[], name='learning_rate_NN')
    ph_lr_fcnet = tf.placeholder(tf.float32,shape=[], name='learning_rate_fcNet')
    
    with tf.name_scope('cross_ent'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=predictions))
    optimizerFCNet = tf.train.AdamOptimizer(learning_rate = ph_lr_fcnet) #0.000001  #optimizer = tf.train.AdamOptimizer(1e-4)
    optimizerNN = tf.train.AdamOptimizer(learning_rate = ph_lr_nn )  #0.00001 #optimizer = tf.train.AdamOptimizer(1e-4)
    with tf.name_scope('train'):
        #train_step = optimizer.minimize(cross_entropy)
        # Op to calculate every variable gradient
        siameseVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='siameseNet_vs')
        diffNetVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='DiffNNet_vs')
        # vars for two sub-nets
        FCNetVars = siameseVars + diffNetVars
        NNVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='NN')
        
        
        grads = tf.gradients(cross_entropy , FCNetVars + NNVars)
        grads1 = grads[:len(FCNetVars)]
        grads2 = grads[len(FCNetVars):]
        train_op1 = optimizerFCNet.apply_gradients(zip(grads1, FCNetVars))
        train_op2 = optimizerNN.apply_gradients(zip(grads2, NNVars))
        train_step = tf.group(train_op1, train_op2)
        #train_step = train_op2
        #grads = tf.gradients(cross_entropy , tf.trainable_variables())
        grads = list(zip(grads, FCNetVars + NNVars))
        # Op to update all variables according to their gradient
        #train_step = optimizerFCNet.apply_gradients(grads_and_vars=grads)        
        #train_step = optimizer.minimize(cross_entropy,name='train')
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(predictions, 1))
    nCorrects_op = tf.count_nonzero(correct_prediction)
    prediction_op = tf.argmax(predictions, 1)
    
    #recall = tf.metrics.recall(labels=tf.argmax(y, 1), predictions=tf.argmax(predictions, 1))
    #precision =  tf.metrics.precision(labels=tf.argmax(y, 1), predictions=tf.argmax(predictions, 1))
    
    with tf.name_scope('accuracySc'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.name_scope("summaries"):
            
            tf.summary.scalar("accuracy",accuracy)
            tf.summary.scalar('Loss',cross_entropy)
            tf.summary.histogram('histogram_loss',cross_entropy)
            tf.summary.histogram('histogram_accuracy',accuracy)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            # Summarize all gradients
            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad)
            summary_op = tf.summary.merge_all()
    
    # model created with loss 
    print ("Number of trainable parameters are {} ".format(getTrainableParamNumber()))
    print("---------------------------------------------------")
    

    print('model created. strating for training...')
    # optional - load data
    matlabFile = 'FCNetExperiment.mat'
    print('Loading '+ datasetName +' dataset ...')
    xTrain = loadMatlabVar(matlabFile,datasetName + 'SubTrainF')
    xTest  = loadMatlabVar(matlabFile,datasetName +'SubTestF')
    phenoTrain = loadMatlabVar(matlabFile,datasetName +'PhenoTrain')
    phenoTest = loadMatlabVar(matlabFile,datasetName +'PhenoTest')
    
    yTrain = extractLabelFromPheno(phenoTrain)
    yTest = extractLabelFromPheno(phenoTest)
    
    # process the data now
    #trainLabelHV = np.array( list(map(lambda x: (0,1) if x==1 else (1,0),yTrain)))
    #testLabelHV = np.array( list(map(lambda x: (0,1) if x==1 else (1,0),yTest)))
    
    trainLabelHV = np.array( list(map(lambda x: (1,0) if x==0 else (0,1),yTrain)))
    testLabelHV = np.array( list(map(lambda x: (1,0) if x==0 else (0,1),yTest)))
    print ('Dataset : Training {}, NOrmal {} ADHD {}'.format(len(yTrain),len(yTrain[yTrain==0]),len(yTrain[yTrain!=0])))    
    print ('Dataset : Test {}, Normal {} ADHD {}'.format(len(yTest),len(yTest[yTest==0]),len(yTest[yTest!=0])))    
    
    
    # soem hyper params
    nSamples = trainLabelHV.shape[0]
    
    nBatches = int(nSamples / batch_size)
    
    xTrain = xTrain.reshape(xTrain.shape[0],xTrain.shape[1],xTrain.shape[2],1)
    xTest = xTest.reshape(xTest.shape[0],xTest.shape[1],xTest.shape[2],1)
    trainStepCount = 0 
    lr_fcnet=0.00001
    lr_nn = 0.0001
    startSess = time.time()
    with tf.Session() as sess:
      # init all vars  
      sess.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter(summaryPath,graph=tf.get_default_graph())
      saver = tf.train.Saver()
      
      for e in range(nEpochs):
          startEpoch = time.time()
          #file = open("{}/results e{}.txt".format(resultsPath,e),"w") 
          loss=0
          train_accuracy =0
          
          for batchNumber in range (0,nBatches):
              #bid*batch_size:(bid+1)*batch_size
              startInd = batchNumber * batch_size
              endInd = (batchNumber + 1) * batch_size
              
    #          if (trainStepCount % 100 == 0):
    #                  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #                  run_metadata = tf.RunMetadata() 
    #                  _,crossLoss,acc,summary = sess.run([train_step,cross_entropy,accuracy,summary_op],
    #                                   feed_dict={x: xTrain[startInd:endInd,:],y: trainLabelHV[startInd:endInd,:]}
    #                                   ,options=run_options, run_metadata=run_metadata)
    #                  writer.add_run_metadata(run_metadata, 'step%d' % trainStepCount)
    #                  writer.add_summary(summary,e * nBatches + batchNumber)
    #          else:
    #                  _,crossLoss,acc,summary = sess.run([train_step,cross_entropy,accuracy,summary_op],
    #                                   feed_dict={x: xTrain[startInd:endInd,:],y: trainLabelHV[startInd:endInd,:]})
    #                  writer.add_summary(summary,e * nBatches + batchNumber)
              _,crossLoss,acc,summary = sess.run([train_step,cross_entropy,accuracy,summary_op],
                                       feed_dict={x: xTrain[startInd:endInd,:],y: trainLabelHV[startInd:endInd,:],
                                                  ph_lr_nn:lr_nn,ph_lr_fcnet:lr_fcnet})
              
              #writer.add_summary(summary,e * nBatches + batchNumber)  
              trainStepCount += 1
              loss += crossLoss
              train_accuracy += acc
              #writer.add_summary(summary,e)        
           #   file.write("e {}, batch id {} accuracy {} loss {} \n".format(e,batchNumber,acc,crossLoss))
          endEpoch = time.time()
          writer.add_summary(summary,e)  
          
          if ( e%10==0):
              testAcc,predictionsVal,corrects = sess.run([accuracy,prediction_op,nCorrects_op],feed_dict={x: xTest,y: testLabelHV})
              spec,sens = getSpecSens(yTest,predictionsVal)
              print('epoch {}, loss {:.2f} training accuracy {:.2f} spec {}, Sens {}, test accuracy {} corrects {}, time {}'.format(e, loss/nBatches,train_accuracy/nBatches,spec,sens,testAcc,corrects,endEpoch - startEpoch))
          else:
              print('epoch %d, loss %g training accuracy %g - time epoch %g' % (e, loss/nBatches,train_accuracy/nBatches,endEpoch - startEpoch))    
          #file.close()
      
      testAcc,predArgMax,predictionsHV = sess.run([accuracy, prediction_op,predictions],feed_dict={x: xTest,y: testLabelHV})
      spec,sens = getSpecSens(yTest,predArgMax)    
      
      print("Time for {} epochs = {} sec".format(nEpochs,time.time() - startSess ))
      print("Test Accuracy : {}, Spec{}, Sens {},  for {} dataset".format(testAcc,spec,sens,datasetName))
      # calculate Sens / Spec 
      print ('generating FC for healthy and ADHD...')
      indicesHealthy = [i for i, x in enumerate(yTrain) if x == 0]
      indicesPatient = [i for i, x in enumerate(yTrain) if x != 0]
      
      trainFC = sess.run(evenVals, feed_dict = {x:xTrain })
      #testFC = sess.run(evenVals, feed_dict = {x:xTest })
      trainFCHealthy = trainFC[indicesHealthy]
      trainFCPatient = trainFC[indicesPatient]
      
    
      save_path = saver.save(sess, ETEModelPath)
      print("Model saved in file: %s" % save_path)   
      
      fileNameToSave = 'FCFromETE'
      #sio.savemat(fileNameToSave,{'trainFC':trainFC,'trainFCHealthy':trainFCHealthy, 'trainFCPatient':trainFCPatient})
	  
      sio.savemat(fileNameToSave,{'yTest':testLabelHV,'predictions':predictionsHV})
	  #print ('FC mats saved in ' + fileNameToSave)
      
      return trainFC, trainFCHealthy, trainFCPatient
	  
    #  print('test accuracy %g' % accuracy.eval(feed_dict={
    #      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        
    
def findInArray(arr, val):
    for i,v in enumerate(arr):
        if (v==val):
            return i;
    return -1

    
    

