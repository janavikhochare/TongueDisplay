from keras.models import model_from_json
from keras.models import load_model

import os, cv2
import numpy as np
##import matplotlib.pyplot as plt
import glob

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as k

k.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from sklearn import preprocessing
# from keras


from keras.preprocessing import image

# images=[]

model = load_model('model1.hdf5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

PATH = os.getcwd()
data_path = PATH + '/image7'
data_dir_list = os.listdir(data_path)

#image.load_img('dataset/training_set/apple_or_football_3.jpg', target_size=(64, 64))
# """
# for dataset in data_dir_list:
#     img_list=os.listdir(data_path+'/'+ dataset)
#     for img in img_list:
#         input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
#         input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
#         input_img_resize=cv2.resize(input_img,(128,128))
#         img_data_list.append(input_img_resize)
#
#
# test_image = image.load_img('dataset/single_prediction/apple_or_football_3.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = model.predict(test_image)
# training_set.class_indices
#
# if result[0][0] == 1:
#     prediction = 'football'
#
# le.fit([tl for tl in data_dir_list])
#model.summary()
count = 1
for i, label in enumerate(data_dir_list):
    cur_path = data_path + '/' + label
    #print(cur_path)
    a = sorted(glob.glob(cur_path))
    j = 1  # type: int
    print(a)

    for image_path in a:

#        img = image.load_img(image_path, target_size=(128,128,1))
        input_img = cv2.imread(image_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (64,64))
        #img_data = np.array(img_data_list)
        img_data = input_img_resize.astype('float32')
        img_data /= 255
       # print(img.shape)
# else:

#     prediction = 'apple'
# print(prediction)
#
# """
# le = preprocessing.LabelEncoder()
       # x = image.img_to_array(img)
        #print (x.shape)
        #x= x.reshape((128,128,1))
        #print(x.shape)
        x = np.expand_dims(img_data, axis=0)
        x = np.expand_dims(x, axis=0)
        x= x.reshape((-1,64,64,1))
        #print(x.shape)
        #assert isinstance(model.predict, object)
        y = model.predict(x)
        print(y)
        #training_set.class_indices
        if y < 0.5:
            prediction = 'apple'  # type: str
        else:
            prediction = 'banana'
        print(prediction)
        print ("[INFO] processed - " + str(j))
        print ("===============================")
    # noinspection PyUnboundLocalVariable
    j += 1
print ('[INFO] completed label - {0}'.format(label))
