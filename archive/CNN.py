import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
# force theano
from keras import backend
backend.set_image_dim_ordering('th')

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

# t-folds
# the first element is a tuple (train_X, test_X)
# the second element is a tuple (train_Y, test_Y)
folds = []
num_folds = 10
subset_size = train_X.shape[0]/num_folds
for i in range(num_folds):
    testingX_this_round = train_X[i*subset_size:][:subset_size]
    print("testingX_this_round", testingX_this_round.shape)
    trainingX_this_round = np.append(train_X[:i*subset_size], train_X[(i+1)*subset_size:], axis=0)
    print("trainingX_this_round", trainingX_this_round.shape)
    testingY_this_round = train_binary_Y[i*subset_size:][:subset_size]
    trainingY_this_round = np.append(train_binary_Y[:i*subset_size],  train_binary_Y[(i+1)*subset_size:], axis=0)
    folds.append(\
            ((trainingX_this_round, testingX_this_round),\
             (trainingY_this_round, testingY_this_round))\
            )
# train with the first fold
train_X_fold = folds[0][0][0]
test_X_fold = folds[0][0][1]
train_Y_fold = folds[0][1][0]
test_Y_fold = folds[0][1][1]

model = Sequential()
# nb size need to be changed
model.add(Convolution3D(32, 3,3,3, subsample=(1, 1, 1), dim_ordering='th',\
            activation='relu',input_shape=(1, 26, 31, 23)))
print model.input_shape
model.add(Convolution3D(32, 3,3,3, subsample=(1, 1, 1), activation='relu'))
model.add(MaxPooling3D(pool_size=(2,2,2)))
model.add(Dropout(0.25))
# fully connected dense layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(19, activation='sigmoid'))
print model.output_shape
# a discussion about which loss function to use
# https://stats.stackexchange.com/questions/207794/what-loss-function-for-multi-class-multi-label-classification-tasks-in-neural-n
model.compile(loss='binary_crossentropy',\
        optimizer='adam',\
        metrics=['accuracy'])
model.fit(train_X_fold, train_Y_fold, \
        batch_size=32, nb_epoch=2, verbose=1)
model.save("./saved_model.h5")
score = model.evaluate(test_X_fold, test_Y_fold, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
