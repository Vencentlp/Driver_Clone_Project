import csv
import cv2
import numpy as np
import os.path
import gc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import Sequential
from keras.layers import Flatten, Dense,Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
import matplotlib.pyplot as plt

# read the data
lines = []
with open ('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = lines[1:]
#print(lines)
#for line in lines:
#    source_path = line[0]
#    file_name = os.path.basename(source_path)
#    current_path = '../data/IMG/'+file_name
#    image = cv2.imread(current_path)
#    images.append(image)
#    measurement = float(line[3])
#    measurements.append(measurement)
#gc.collect()


# augment data
#aug_images=[]
#aug_measurements=[]
#for image, measurement in zip(images,measurements):
#    aug_images.append(image)
#    aug_measurements.append(measurement)
#    aug_images.append(cv2.flip(image,1))
#    aug_measurements.append(measurement*-1)
#gc.collect()

# test and validation data split



# define function of batch generator

def generator (samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                current_path = '../data/IMG/'
                path_center = current_path+os.path.basename(batch_sample[0])
                print(path_center)
                path_left = current_path+os.path.basename(batch_sample[1])
                path_right = current_path+os.path.basename(batch_sample[2])
                correction = 0.2
                image_center = cv2.imread(path_center)
                measurement_center = float(batch_sample[3])
                measurement_left = measurement_center + correction
                measurement_right = measurement_center - correction
                image_left = cv2.imread(path_left)
                image_right = cv2.imread(path_right)
                image_flip = cv2.flip(image_center, 1)
                measurement_flip = measurement_center*-1
                images.extend([image_center, image_flip, image_left, image_right])
                measurements.extend([measurement_center, measurement_flip, measurement_left, measurement_right])
            x_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(x_train, y_train)


# X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.2)
train_samples, valid_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
valid_generator = generator(valid_samples, batch_size=32)
gc.collect()


# define the Deep Learning pipeline
model = Sequential()
model.add(Lambda(lambda x: x/255-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation=('relu')))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation=('relu')))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation=('relu')))
model.add(Convolution2D(64,3,3,activation=('relu')))
model.add(Convolution2D(64,3,3,activation=('relu')))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Train and vlidation

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                     validation_data=valid_generator,nb_val_samples=len(valid_samples),
                                     nb_epoch=3, verbose=1, pickle_safe=False)
model.save('model.h5')

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

