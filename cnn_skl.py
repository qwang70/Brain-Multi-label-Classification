import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.utils import np_utils
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
# force theano
from keras import backend
import tensorflow as tf
sess = tf.Session()
backend.set_session(sess)
backend.set_image_dim_ordering('tf')

# Function to create model, required for KerasClassifier
def create_model():
    # code piece used to train on whole dataset
    model = Sequential()
    # nb size need to be changed
    model.add(Conv3D(8, (3,3,3), strides=(1, 1, 1), activation='relu', data_format="channels_first", input_shape=(1, 26, 31, 23)))

    model.add(Conv3D(filters=8, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu'))
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
            metrics=[accuracy_score])
    return model
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
    # use this piece of script when training & testing on known X, Y

    # t-folds
    # create model
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=10, verbose=1)
    # evaluate using 10-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = cross_val_score(model, train_X, train_binary_Y, cv=kfold)
    print(results)
    print(results.mean())

    """
    model = create_model
    model.fit(train_X, train_binary_Y, \
            batch_size=8, epochs=20, verbose=1)
    model.save("./saved_model.h5")
    """

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
