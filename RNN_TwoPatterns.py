from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import argparse
import os
import theano
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from theano import tensor
from keras import backend as K
from keras.layers import Bidirectional
from sklearn import svm
from sklearn.svm import SVC

np.random.seed(1234)


def build_model():
    model = Sequential()
    layers = [1, 24, 24, 1]
    
    model.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
    model.add(Dropout(0.1))
    
    model.add(LSTM(output_dim=layers[2],return_sequences=False))
    model.add(Dropout(0.1))
    
    model.add(Dense(output_dim=layers[3]))
    #model.add(Activation("tanh"))
    
    start = time.time()
    model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])
    print "Model Compilation Time : ", time.time() - start
    return model


def run_network(model=None, X_train=None, X_label=None, X_test=None, X_test_label=None):
    
    print "Starting SVM ground truth:"
    clf_gt=SVC()
    x_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    print "Training..."
    clf_gt.fit(x_train,X_label)
    print "Complete."
    print "Starting SVM ground truth predicting:"
    x_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    predict_labels_gt=clf_gt.predict(x_test)
    #np.savetxt('FacesUCR_test_labels.txt', predict_labels_gt, fmt='%s')
    print predict_labels_gt
    count_1=0
    for x in range(0,X_test_label.shape[0]):
        if X_test_label[x]==predict_labels_gt[x]:
            count_1+=1
    print "Count: ",count_1
    print "Accuracy: ",count_1/X_test_label.shape[0]
    
    global_start_time = time.time()
    epochs = 200
    
    if model is None:
        model = build_model()
    
    try:
        model.fit(X_train, X_label,batch_size=25, nb_epoch=epochs)#, validation_split=0.05
    except KeyboardInterrupt:
        print 'Training duration (s) : ', time.time() - global_start_time
        return model, 0
    print 'Training duration (s) : ', time.time() - global_start_time

    get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[0].output])
    layer_output = get_3rd_layer_output([X_train, 1])[0]
    layer_output=np.reshape(layer_output, (1,layer_output.shape[0],layer_output.shape[1]*layer_output.shape[2]))
    layer_output=layer_output[0]
    print "TRAIN OUTPUT: ",layer_output
    print "Train Output Shape 0: ", layer_output.shape[0]
    print "Train Output Shape 1: ", layer_output.shape[1]
    #print "Train Output Shape 2: ", layer_output.shape[2]
    np.savetxt('TwoPatterns_train_24D.txt', layer_output, fmt='%s')
    
    eval_loss,eval_accuracy=model.evaluate(X_test,X_test_label,batch_size=2)
    #evaluate = np.reshape(evaluate, (evaluate.size,))
    print "Scalar test loss: ",eval_loss
    print "Accuracy: ",eval_accuracy
    
    #predicted = model.predict(X_test)
    #predicted = np.reshape(predicted, (predicted.size,))
    
    
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[0].output])
    layer_output_after = get_layer_output([X_test, 0])[0]
    layer_output_after=np.reshape(layer_output_after, (1,layer_output_after.shape[0],layer_output_after.shape[1]*layer_output_after.shape[2]))
    layer_output_after=layer_output_after[0]
    print "TEST OUTPUT PREDICT: ",layer_output_after
    print "TEST OUTPUT PREDICT: ",layer_output_after
    #print "Predict Output Shape 0: ", layer_output_after.shape[0]
    #print "Predict Output Shape 1: ", layer_output_after.shape[1]
    #print "Predict Output Shape 2: ", layer_output_after.shape[2]
    np.savetxt('TwoPatterns_test_24D.txt', layer_output_after, fmt='%s')
    
    print "Starting SVM:"
    clf=SVC()
    print "Training..."
    #print "SVM TRAIN: ",layer_output
    print "SVM TRAIN LABELS: ",X_label
    #print "SVM TEST: ",layer_output_after
    clf.fit(layer_output,X_label)
    print "Complete."
    print "Starting SVM predicting:"
    predict_labels=clf.predict(layer_output_after)
    np.savetxt('TwoPatterns_test_labels_predict.txt', predict_labels, fmt='%s')
    print predict_labels
    count=0
    for x in range(0,X_test_label.shape[0]):
        if X_test_label[x]==predict_labels[x]:
            count+=1
    print "Count: ",count
    print "Accuracy: ",count/X_test_label.shape[0]
    return model




parser = argparse.ArgumentParser()
parser.add_argument('train_dir', help='directory with txt files')
parser.add_argument('test_dir', help='directory with txt files')
args = parser.parse_args()
train_dir = args.train_dir
test_dir = args.test_dir
train_filenames = os.listdir(train_dir)
test_filenames = os.listdir(test_dir)
train_timeseries = []
test_timeseries = []

print "Reading training files from", train_dir
for filename in train_filenames:
    ts = np.genfromtxt(os.path.join(train_dir, filename))
    if ts.ndim == 1:  # csv file has only one column, ie one variable
        ts = ts[:, np.newaxis]
    train_timeseries.append(ts)
train_timeseries = np.array(train_timeseries)
train_num_timeseries, train_num_timesteps, train_num_dims = train_timeseries.shape
print "Read {} training time series with {} time steps and {} dimensions".format(train_num_timeseries, train_num_timesteps, train_num_dims)
#print "TRAINING TIME SERIES: ", train_timeseries

print "Reading testing files from", test_dir
for filename in test_filenames:
    ts = np.genfromtxt(os.path.join(test_dir, filename))
    if ts.ndim == 1:  # csv file has only one column, ie one variable
        ts = ts[:, np.newaxis]
    test_timeseries.append(ts)
test_timeseries = np.array(test_timeseries)
test_num_timeseries, test_num_timesteps, test_num_dims = test_timeseries.shape
print "Read {} testing time series with {} time steps and {} dimensions".format(test_num_timeseries, test_num_timesteps, test_num_dims)
#print "TESTING TIME SERIES: ", test_timeseries

#num_train=round(train_timeseries.shape[0])
#num_test=round(test_timeseries.shape[0])

X_train=train_timeseries[:,:,1:train_timeseries.shape[2]]
X_label=train_timeseries[:,:,[0]]
X_test=test_timeseries[:,:,1:train_timeseries.shape[2]]
X_test_label=test_timeseries[:,:,[0]]
#print "Shape0: ",X_train.shape[0]
#print "Shape1: ",X_train.shape[1]
#print "Label Shape0: ",X_label.shape[0]
#print "Label Shape1: ",X_label.shape[1]
#print "Train data: ",X_train


X_train=np.reshape(X_train, (X_train.shape[0]*X_train.shape[1], X_train.shape[2],1))
X_test=np.reshape(X_test, (X_test.shape[0]*X_test.shape[1], X_test.shape[2],1))
X_label=np.reshape(X_label, (1,X_label.shape[0]*X_label.shape[1]))
X_test_label=np.reshape(X_test_label, (1,X_test_label.shape[0]*X_test_label.shape[1]))
X_label=X_label[0]
X_test_label=X_test_label[0]
np.savetxt('TwoPatterns_train_labels_truth.txt', X_label, fmt='%s')
np.savetxt('TwoPatterns_test_labels_truth.txt', X_test_label, fmt='%s')
#print "Train Label: ",X_label
#print "Test data: ",X_test
#print "Test Label: ",X_test_label
#np.savetxt('RNN_EEG_train_labels.txt', X_label, fmt='%s')
#np.savetxt('RNN_EEG_test_labels.txt', X_test_label, fmt='%s')
print "TRAIN TIME SERIES: ", X_train
print "TRAIN Data Shape 0: ", X_train.shape[0]
print "TRAIN Data Shape 0: ", X_train.shape[1]
print "TRAIN Data Shape 0: ", X_train.shape[2]
print "TEST TIME SERIES: ", X_test
print "TEST Data Shape 0: ", X_test.shape[0]
print "TEST Data Shape 0: ", X_test.shape[1]
print "TEST Data Shape 0: ", X_test.shape[2]

run_network(None,X_train,X_label,X_test,X_test_label)
