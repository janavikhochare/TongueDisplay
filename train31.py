# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation,BatchNormalization, Dropout, Flatten, Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import os,cv2


from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

#from keras.models import load_model
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
from numpy.core.multiarray import ndarray

epochs =10

img_data_list=[]
labels_list=[]
#labels_list[1900]=0
#labels_list[1900:]=1


PATH = os.getcwd()
# Define data path
data_path = PATH + '/dataset3'
data_dir_list = os.listdir(data_path)

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loading the images of dataset-'+'{}\n'.format(dataset))
	#label = labels_name[dataset]
	for img in img_list:
	#	print(img)
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(64,64))
		if img[0]=='a':
			labels_list.append('0')
		else:
			labels_list.append('1')

		img_data_list.append(input_img_resize)
		#labels_list.append(label)
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)
#print(labels_list)

labels = np.array(labels_list)
# print the count of number of samples for different classes
print(np.unique(labels,return_counts=True))
# convert class labels to on-hot encoding
#Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,labels, random_state=4)
x=np.expand_dims(x,axis=1).reshape(-1,64,64,1)

# Split the dataset
training_set,test_set,train_label,test_label = train_test_split(x, y, test_size=0.2, random_state=2)

"""
model = Sequential()
model.add(Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(64,64,1)))
#model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())
#model.add(Dropout(0.1))
#model.add(Dropout(0.25))
 
model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
#model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())
#model.add(Dropout(0.1))
#model.add(Dropout(0.25))
 
model.add(Conv2D(128, (5,5), padding='same', activation='relu'))
#model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
 
model.add(Flatten())
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))
"""

#loaded_model=load_model('model.hdf5')
# Initialising the CNN

model = Sequential()

# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape = (64,64,1), activation = 'relu'))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# Adding a second convolutional layer(changed to 64)
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.15))
#i added a convolution layer cos accuracy wasnt changing
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.2))
# Step 4 - Full connection
model.add(Dense(units = 128, activation = 'relu' ))
model.add(Dense(units = 1, activation = 'sigmoid'))

# label=np.zeros((800,1))
# label[400:]=1
#
# test_label=np.zeros((200,1))  # type: ndarray
# test_label[100:]=1
# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

# from keras.preprocessing.image import ImageDataGenerator
#
# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)
#
# test_datagen = ImageDataGenerator(rescale = 1./255)
#
# training_set = train_datagen.flow_from_directory('dataset/training_set',
#                                                  target_size = (64, 64),
#                                                  batch_size = 100,
#                                                  class_mode = 'binary')
#
# test_set = test_datagen.flow_from_directory('dataset/test_set',
#                                             target_size = (64, 64),
#                                             batch_size = 25,
#                                             class_mode = 'binary')
"""
for mini_batch in range(epochs):
        model_hist = model.fit(training_set,train_label, batch_size=100, epochs=100,
                            verbose=2, validation_data=(test_set,test_label))

        precision = model_hist.history['val_precision'][0]
        recall = model_hist.history['val_recall'][0]
        f_score = (2.0 * precision * recall) / (precision + recall)
        print 'F1-SCORE {}'.format(f_score)
"""
model.fit(training_set,train_label,batch_size=50,epochs = 10,validation_data = (test_set,test_label))
# from keras import callbacks
#
#
# filename='model_train_new.csv'
# csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
#
# early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')
#
# filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
#
# checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#
# callbacks_list = [csv_log,early_stopping,checkpoint]
#
# hist = model.fit_generator(training_set,label,
#                          steps_per_epoch = 8,
#                          epochs = 10,
#                          validation_data = (test_set,test_label),validation_steps = 200,callbacks=callbacks_list)

# train_loss=hist.history['loss']
# val_loss=hist.history['val_loss']
# train_acc=hist.history['acc']
# val_acc=hist.history['val_acc']
# xc=range(epochs)

# Evaluating the model
"""
score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])

# Part 3 - Making new predictions
# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('dataset/single_prediction/apple_or_football_3.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = model.predict(test_image)
# tra
#
# if result[0][0] == 1:
# 	prediction = 'football'
#
# else:
# 	prediction = 'apple'
# print(prediction)
#

# Saving and loading model and weights

"""
# serialize model to JSON
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
print("Loaded model from disk")

model.save('model1.hdf5')
loaded_model=load_model('model1.hdf5')
#training_set.save('train3.h5')
print("done")
#activity_regularizer=regularizers.l1(0.01)
