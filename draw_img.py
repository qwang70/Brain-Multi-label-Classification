import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_X = np.load('train_X.npy')
print(train_X.shape)

print(train_X[0,1,1])
print(train_X[0,2,1])
print("mu",np.mean(train_X[0,1]))
print("std",np.std(train_X[0,1]))
"""
for idx in range(10):
    fig = plt.figure()
    for i in range(25):
        print(i)
        plt.subplot(5,5,i+1)
        imgplot = plt.imshow(train_X[idx,i])
        fig.savefig("brain{}".format(idx))
plt.show()
"""
