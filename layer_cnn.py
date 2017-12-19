import itertools
import operator
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv3D, MaxPooling3D
from keras.utils import np_utils
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from subprocess import check_output



from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator




import matplotlib.pyplot as plt
# force theano
from keras import backend
import tensorflow as tf
sess = tf.Session()
backend.set_session(sess)
backend.set_image_dim_ordering('tf')


# Function to create model, required for KerasClassifier
def create_model(filters = (24,24)):
    # code piece used to train on whole dataset
    model = Sequential()
    # nb size need to be changed
    model.add(Conv3D(64, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu', data_format="channels_last", input_shape=( 26, 31, 23, 1)))

    #model.add(Conv3D(filters = 32, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu'))
    model.add(MaxPooling3D(pool_size=(3,3,3)))
    model.add(BatchNormalization())
    model.add(Conv3D(filters = 128, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu'))
    model.add(Dropout(0.25))
    #model.add(Conv3D(filters = 64, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu'))
    #model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Conv3D(filters = 256, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu'))
    #model.add(Conv3D(filters = 128, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu'))
    model.add(Dropout(0.25))
    # fully connected dense layer
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    # or a smaller dense
    model.add(Dense(45, activation='softmax'))

    # a discussion about which loss function to use
    # https://stats.stackexchange.com/questions/207794/what-loss-function-for-multi-class-multi-label-classification-tasks-in-neural-n
    # they're using SGD
    # https://stackoverflow.com/questions/44164749/how-does-keras-handle-multilabel-classification
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics = ["accuracy"])
    print(model.summary())
    return model

# load and reshape input data
tag_name = np.load('tag_name.npy')
#(19,)
train_X = np.load('train_X.npy')
#(4602, 26, 31, 23)
train_X = train_X.reshape(train_X.shape[0], \
        train_X.shape[1], train_X.shape[2], train_X.shape[3], 1)
train_X = train_X.astype('float32')
# (4602, 26, 31, 23, 1)
train_binary_Y = np.load('train_binary_Y.npy')
np.random.seed(123)  # for reproducibility

possible_comb = {}
augmented_Y = np.zeros((len(train_X), 45))
class_num = 0
for i in range(len(train_binary_Y)):
    data = train_binary_Y[i]
    flag = True
    for j in possible_comb.keys():
        a = possible_comb[j] == data
        if a.all():
            flag = False
            break
    if flag == True:
        possible_comb[class_num] = data
        class_num = class_num + 1

for j in range(len(train_binary_Y)):
    data = train_binary_Y[j]
    for i in possible_comb.keys():
        a = data == possible_comb[i]
        if a.all():
            augmented_Y[j, i] = 1
            break

#(4602, 19)
valid_test_X = np.load('valid_test_X.npy')
#(1971, 26, 31, 23)

valid_test_X = valid_test_X.reshape(\
        valid_test_X.shape[0], \
        valid_test_X.shape[1], valid_test_X.shape[2], valid_test_X.shape[3], 1)
#(1971, 26, 31, 23, 1)
valid_test_X = valid_test_X.astype('float32')

saved_model = "saved_model.h5"

# shuffle training data
print(train_X.shape, train_binary_Y.shape)

model = create_model()
history = model.fit(train_X, augmented_Y,
                    epochs=100,
                    batch_size=50,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=2)
# list all data in history
print(history.history.keys())

# use trained model by adding any other paramerers in input

#test_data = train_X
test_data = valid_test_X
pred = model.predict(test_data, verbose=1)
"""
prediction = np.zeros((len(pred), 19))
for i in range(len(pred)):
    for j in range(45):
        label = possible_comb[j]
        for k in range(19):
            if label[k] == 1:
                prediction[i, k] = prediction[i,k] + pred[i, j]
"""
'''
assign the largest possibility combination as the label class
'''
labels=[]
pred_labels=np.zeros((len(pred), 19))
for i in range(len(pred)):
    labels.append(np.argmax(pred[i]))
for i in range(len(labels)):
    pred_labels[i]=possible_comb[labels[i]].T

#np.save("li_result.npy", prediction)
np.save("li_label_result.npy", pred_labels)
