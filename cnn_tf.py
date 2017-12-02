import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.utils import np_utils
from keras.models import load_model
# force theano
from keras import backend
import tensorflow as tf
sess = tf.Session()
backend.set_session(sess)
backend.set_image_dim_ordering('tf')

# load and reshape input data
tag_name = np.load('tag_name.npy')
#(19,)
train_X = np.load('train_X.npy')
#(4602, 26, 31, 23)
train_X = train_X.reshape(train_X.shape[0], 1, \
        train_X.shape[1], train_X.shape[2], train_X.shape[3])
train_X = train_X.astype('float32')
#(4602, 1, 26, 31, 23)
train_binary_Y = np.load('train_binary_Y.npy')
#(4602, 19)
valid_test_X = np.load('valid_test_X.npy')
#(1971, 26, 31, 23)
valid_test_X = valid_test_X.reshape(\
        valid_test_X.shape[0], 1, \
        valid_test_X.shape[1], valid_test_X.shape[2], valid_test_X.shape[3])
#(1971, 1, 26, 31, 23)
valid_test_X = valid_test_X.astype('float32')

saved_model = "saved_model.h5"
if len(sys.argv) == 1:
    # train model
    print("train model")

    
    """
    # use this piece of script when training & testing on known X, Y

    # t-folds
    # the first element is a tuple (train_X, test_X)
    # the second element is a tuple (train_Y, test_Y)
    folds = []
    num_folds = 5
    subset_size = int(train_X.shape[0]/num_folds)
    for i in range(num_folds):
        testingX_this_round = train_X[i*subset_size:][:subset_size]
        trainingX_this_round = np.append(train_X[:i*subset_size], train_X[(i+1)*subset_size:], axis=0)
        testingY_this_round = train_binary_Y[i*subset_size:][:subset_size]
        trainingY_this_round = np.append(train_binary_Y[:i*subset_size],  train_binary_Y[(i+1)*subset_size:], axis=0)
        folds.append(\
                ((trainingX_this_round, testingX_this_round),\
                 (trainingY_this_round, testingY_this_round))\
                )

    # train & test on each fold
    scores = []
    for i in range(num_folds):
        # train with the first fold
        print ("train on fold set {}".format(i))
        train_X_fold = folds[i][0][0]
        test_X_fold = folds[i][0][1]
        train_Y_fold = folds[i][1][0]
        test_Y_fold = folds[i][1][1]

        model = Sequential()
        # nb size need to be changed
        model.add(Conv3D(32, (3,3,3), strides=(1, 1, 1), activation='relu', data_format="channels_first", input_shape=(1, 26, 31, 23)))

        print (model.input_shape)
        model.add(Conv3D(filters=32, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu'))
        model.add(MaxPooling3D(pool_size=(2,2,2)))
        model.add(Dropout(0.25))
        # fully connected dense layer
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(19, activation='sigmoid'))
        print (model.output_shape)
        # a discussion about which loss function to use
        # https://stats.stackexchange.com/questions/207794/what-loss-function-for-multi-class-multi-label-classification-tasks-in-neural-n
        model.compile(loss='binary_crossentropy',\
                optimizer='adam',\
                metrics=['accuracy'])
        model.fit(train_X_fold, train_Y_fold, \
                batch_size=32, epochs=2, verbose=1)

        print ("test on fold set {}".format(i))
        score = model.evaluate(test_X_fold, test_Y_fold, verbose=0)
        scores.append(score)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        model.save("./saved_model.h5")
    scores = np.array(scores)
    print(scores)
    print("Average score:\t{}".format(np.average(scores)))
    

    """
    # code piece used to train on whole dataset
    model = Sequential()
    # nb size need to be changed
    model.add(Conv3D(32, (3,3,3), strides=(1, 1, 1), activation='relu', data_format="channels_first", input_shape=(1, 26, 31, 23)))

    print (model.input_shape)
    model.add(Conv3D(filters=32, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Dropout(0.25))
    # fully connected dense layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(19, activation='sigmoid'))
    print (model.output_shape)
    # a discussion about which loss function to use
    # https://stats.stackexchange.com/questions/207794/what-loss-function-for-multi-class-multi-label-classification-tasks-in-neural-n
    # they're using SGD
    # https://stackoverflow.com/questions/44164749/how-does-keras-handle-multilabel-classification 
    model.compile(loss='binary_crossentropy',\
            optimizer='adam',\
            metrics=['accuracy'])
    model.fit(train_X, train_binary_Y, \
            batch_size=32, epochs=20, verbose=1)
    model.save("./saved_model.h5")

else:
    # use trained model by adding any other paramerers in input
    print("use saved model \"{}\"".format(saved_model))
    model = load_model(saved_model)
    # test_data = train_X[:5]
    test_data = valid_test_X
    pred = model.predict(test_data, verbose=1)
    prediction = np.round(pred)
    np.save("result.npy", prediction)
    
    """
    for i in range(len(prediction)):
        # code piece used for data exploration 
        # i.e. compare prediction results on known X, Y
        # choose test_data be train_X[:10] for example

        print( "test sample {}".format(i) )
        # print ("raw prediction:\t {}".format( pred[i]) )
        print ("prediction:\t {}".format( prediction[i].astype(int)) )
        print ("actual:    \t {}".format( train_binary_Y[i]) )
    """
