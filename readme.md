# Goal: Implement a Handwritten Digit Recognition Classifier using CNN in Keras

### Beginning Trials 

Starting Model: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

* Changed data = (x_test, y_test) to validation_split=1/6 in the original model **Accuracy: 0.9911**
* epoch = 15, kernel size = (5,5), batch_size = 256, Dense = 512, PoolSize = (3,3) **Accuracy: 0.9936**
* epoch = 15, kernel size = (5,5), batch_size = 128, + batchnormalizer **Accuracy: 0.9938**
* epoch = 15, kernel size = (5,5), batch_size = 256, Dense = 256 **Accuracy: 0.994**
* Ran the model with additional layers for 15 epochs **Accuracy: 0.9949**


### For our best model, we get accuracy of 0.9957. Here is the detailed explanation of our code logic.


import keras                                                      # import keras <br/>
from keras.datasets import mnist               # import mnist dataset <br/>
from keras.models import Sequential        # import Sequetial model <br/>
from keras.layers import Dense, Dropout, Flatten, BatchNormalization     # import these layers <br/>
from keras.layers import Conv2D, MaxPooling2D      # import layers <br/>
from keras import backend as K                    # import backend <br/>
<br/>
batch_size = 128       # set batch_size to be 128 <br/>
num_classes = 10     # set number of classes to be 10 <br/>
epochs = 25                # set epochs to be 25 <br/>
<br/>
img_rows, img_cols = 28, 28    # The input image dimensions are 28 by 28 <br/>
<br/>
(x_train, y_train), (x_test, y_test) = mnist.load_data()   # load data and split between train and test dataset <br/>
                                                                                                      
<br/><br/>
if K.image_data_format() == 'channels_first': <br/>
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols) <br/>
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols) <br/>
    input_shape = (1, img_rows, img_cols) <br/>
<br/>
*# if  K.image_data_format() is "channels_first", reshape x_train data to have the dimensions: #number of rows of x_train, 1,* <br/> *# img_rows which iw 28, img_cols which is 28* <br/>
*# reshape x_test data to have the dimensions:* <br/>
*# number of rows of x_test, 1, 28,28 . here 28 is img_rows and also img_cols* <br/>
<br/>
else: <br/>
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) <br/>
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) <br/>
   <br/>
 input_shape = (img_rows, img_cols, 1) <br/>

<br/><br/>
*# If  K.image_data_format() is not "channels_first", reshape x_train to have the dimensions:* <br/>
*# number of rows of x_train, 28,28,1* <br/>
*# reshape x_test to have the dimensions:* <br/>
*# number of rows of x_test, 28,28,1.* <br/>
*# here 28 is the  img_rows and img_cols we defined above.* <br/>

<br/><br/>
x_train = x_train.astype('float32')       # reset the data type of x_train to be 'float32' <br/>
x_test = x_test.astype('float32')           # reset the test data type pf x_test to be 'float32' <br/>
x_train /= 255    # Divides x_train with 255 and assign the result to x_train <br/>
x_test /= 255      # Divides x_test with 255 and assign the result to x_test <br/>
print('x_train shape:', x_train.shape)    # print x_train shape <br/>
print(x_train.shape[0], 'train samples') # print x_train row numbers <br/>
print(x_test.shape[0], 'test samples')   # print x_test row numbers <br/>
<br/>
*# convert class vectors to binary class matrices* <br/>
y_train = keras.utils.to_categorical(y_train, num_classes)   # convert y_tarin vectors to binary class matrices. <br/>
y_test = keras.utils.to_categorical(y_test, num_classes) # convert y_test vectors to binary class matrices. <br/>
<br/>
model = Sequential()     # set up the basic model " Sequential". <br/>
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))   <br/>
<br/>
*# add Conv2D layer：Set filter to be 32, which means the number of output filters in the convolution # is 32. Set kernel size to be 5 by 5, which is the width and height of the 2D convolution window.* <br/>
*# Set the activation to be ‘relu’ and set the input_shape to be the shape we defined just now.* <br/>
<br/>
model.add(BatchNormalization()) <br/>
*# add a batch normalization layer* <br/>
<br/>
model.add(Conv2D(64, (5, 5), activation='relu')) <br/>
*# Add a Conv2D layer with filter of 64 and kernel size of 5 by 5. Use the activation ‘relu’.* <br/>
<br/>
model.add(BatchNormalization()) <br/>
*# Add a batch normalization layer* <br/>
<br/>
model.add(Conv2D(64, (5, 5), activation='relu')) <br/>
*# Add a Conv2D layer with filter of 64 and kernel size of 5 by 5. Use the activation ‘relu’.* <br/>
<br/>
model.add(BatchNormalization()) <br/>
*# Add a batch normalization layer* <br/>
<br/>
model.add(MaxPooling2D(pool_size=(3, 3))) <br/>
*# add a maxpooling2D layer with pool size of 3 by 3.* <br/>
<br/>
model.add(Dropout(0.25)) <br/>
*# add a dropout layer with the rate of 0.25, which means we set 25% of the input units to drop.* <br/>

<br/><br/>
model.add(Flatten()) <br/>
*# add a flatten layer* <br/>
<br/>
model.add(Dense(256, activation='relu')) <br/>
*# add a dense layer with dimensionality of the output space to be 256 and use the activation to be ‘relu’.* <br/>
<br/>
model.add(Dropout(0.5)) <br/>
*# add a dropout layer with the rate of 0.5, which means we set 50% of the input units to drop.* <br/>
<br/>
model.add(Dense(256, activation='relu')) <br/>
*# add a dense layer with dimensionality of the output space to be 256 and use the activation to be ‘relu’.* <br/>
<br/>
model.add(Dropout(0.5)) <br/>
*# add a dropout layer with the rate of 0.5, which means we set 50% of the input units to drop.* <br/>
<br/>
model.add(Dense(num_classes, activation='softmax')) <br/>
*# add a dense layer with dimensionality of the output space to be number of class we set earlier,  and use the activation to be ‘softmax’.* <br/>
<br/>
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy']) <br/>
*# compile the model* <br/>
<br/>
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1) <br/>
*# fit the model* <br/>
<br/>
score = model.evaluate(x_test, y_test, verbose=1) <br/>
*# calculate the prediction accuracy score when test the model on test dataset.* <br/>
<br/>
print('Test loss:', score[0]) <br/>
print('Test accuracy:', score[1]) <br/>
<br/>
*# print test loss and test accuracy.* <br/>





