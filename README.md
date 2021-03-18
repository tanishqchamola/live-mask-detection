# Live Mask Detection using Convolutional Neural Network
_Puneet, Tanishq Chamola, Siddharth Samber_<br />
_**CSE 7th semester, Chandigarh College of Engineering and Technology**_

_Dataset - [https://github.com/balajisrinivas/Face-Mask-Detection/](https://github.com/balajisrinivas/Face-Mask-Detection/)_

*Abstract: Face mask recognition has made considerable strides in the fields of computer vision and image processing. Several algorithms and classifiers have been used to construct a variety of face recognition models. We're using deep learning, Tensorflow, Keras, and opencv to train our covid-19 mask detector. This model has the potential to be useful in the area of safety. We have trained our model using Convolutional Neural Network and then we have tested our model on a dataset of static images and finally we have used that model for live face mask detection using a webcam.*<br />
*So basically this project has been divided into two parts, in the training part mask detection model will be trained using the dataset provided. and in the application part we will load our trained model to perform mask detection. We obtained training accuracy of about 99.49%. We will be compiling our model using Adam Optimizer and we will use categorical Cross Entropy as a loss function. The output images are preprocessed for removing unwanted errors and will make a rectangle box around the mask or the face.*

## 1. Introduction

Since December 2019, the COVID-19 pandemic has wreaked havoc in a variety of countries around the world. Wuhan, China, is the place where it all began. The World Health Organization (WHO) declared a deadly disease on March 11, 2020, after it spread across the globe and impacted 114 countries seriously. Due to Covid-19 pandemic people have to wear face masks. Wearing a mask during this pandemic is a crucial prevention measure, and it's especially important in moments when maintaining social distance is difficult.

In the field of image recognition and computer vision, face mask identification has proven to be a difficult challenge. Face mask detection technology appears to be well addressed due to rapid advancements in the domain of machine learning algorithms. This technology is more important today because it is used to identify faces in real-time observation and supervision as well as in static photographs and videos.Image recognition and target identification can now be performed with high precision due to developments in convolutional neural networks and deep learning. Therefore we have trained a CNN which help us to detect whether person has wore mask or not.

## 2. Tools
#### 2.1 Technology
**Python 3:** It is a programming language with many in-built libraries for deep learning, computer vision and so many other applications that makes building models easier.<br />
**OpenCV:** It is a python library with many programming functions with the aim to provide tools which help solve the real time computer vision problems easily. The OpenCV library contains both the low-level image-processing algorithms and high-level functions such as face detection, feature matching, pedestrian detection, and tracking.<br />
**Keras:** It is an open source library in python used for building neural networks. It reduces the number of user actions that are needed for common use cases, and it gives clear, comprehensible and actionable error messages. This library is mainly used for either convolutional networks or recurrent networks or a combination of both.<br />
**os:** This module provides various functions required to interact with the operating system. These functions work independently of the operating system type; all these functions work the same way on all the platforms.<br />
#### 2.2 Dataset
The source of our dataset is a github repository which is at [https://github.com/balajisrinivas/Face-Mask-Detection/](https://github.com/balajisrinivas/Face-Mask-Detection/). Our dataset consist of 3845 images out of which 1916 images are with mask 1929 images are without mask.

| Category      | Without Mask |
| ------------- |:-------------:|
| With Mask     | 1916     |
| Without Mask  | 1929     |

## 3. Method
#### 3.1 Data preprocessing
We are first converting all Images to grayscale and resizing them to 100x100 and their corresponding labels whether with mask or without mask with the help of dictionary. Then We are Normalizing the pixel Values in the range [0,1] from [0,255].

```
img_size=100
data=[]
target=[]

for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
            resized=cv2.resize(gray,(img_size,img_size))
            data.append(resized)
            target.append(label_dict[category])

        except Exception as e:
            print('Exception:',e)
```

#### 3.2 Training CNN (model making)
The generic layers used in a CNN model are:
**Input Layer:** This is the first layer used in a CNN. This layer is used to take an image as input; also to instantiate a Keras tensor, the input layer is used.<br />
**Convolutional Layer:** This layer is also known as the kernel. It is the foundation layer of CNN. The layer which extracts the input features from the image. There can be various convolutional layers with the first layer extracting the low level features from the image and the consecutive layers extracting the high level features.<br />
**Pooling Layer:** It is a form of non-linear downsampling; performing dimensionality reduction, it reduces the computational power required to process the data. There are two types of pooling: Average Pooling and Max Pooling. The Max Pooling layer returns the maximum value from the image part covered by the convolutional layer. This layer performs dimensionality reduction and denoising, and hence creates feature maps that summarize all the input features.<br />
**Flatten Layer:** The pooled feature map obtained from the max pooling layer is flattened in this layer, i.e. converting it into a long vector suitable enough to be easily processed further by the artificial neural network, thus making back propagation easier.<br />
**Dense Layer:** The dense layer computes the dot product between the input and the kernel along the last axis of the inputs and axis 1 of the kernel.<br />
**Dropout Layer:** This layer is mainly a regularization technique; it is added to prevent the over-fitting of the model. An over fitted model has an extended capacity to learn the noise in the observation and as a result poor predictions are made by the model outside the domain of the training set.<br />

```
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

model=Sequential()

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
```

**Layers Used In our Convolutional Model:**
* _Convolutional Layer:_ There are 200 filters and dimension of Kernel is (3,3).
* _Activation Layer:_ (ReLU)The rectified linear activation function is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.
* _Maxpooling Layer:_ Pool Size of this layer is (2,2).
* _Convolutional Layer:_ There are 100 filters and dimension of Kernel is (3,3).
* _Activation Layer:_ (ReLU)The rectified linear activation function is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.
* _Maxpooling Layer:_ Pool Size of this layer is (2,2).
* _Flatten:_ This Layer flattens the input.
* _Dropout:_ Rate of dropout layer is 0.5
* _Dense:_ Dimensionality of output space is 50 and activation is ReLU.
* _Dense:_ Dimensionality of output space is 2 and activation is Softmax.

#### 3.3 Splitting dataset and Accuracy of model
We have split the dataset in 10% test data and 90% training data.We have compile the model and apply it using fit function with 20 epochs and validation split of 0.2. Then we have plot the graphs for accuracy and loss. We got validation accuracy of 91.86% and training accuracy of 98.90%.

#### 3.4 Testing on openCV
We have used Haarcascade Classifier for face detection. For this we are using the file haarcascade_frontalface_default.xml. We are taking (frames) images from video feed and converting it to grayscale and then by using Haarcascade classifier we are detecting the face portion and then we are resizing and normaling that face portion and giving it to our model for detecting whether face is with mask or not. We are showing a red box around the face if the person is not wearing a mask and a green box if that person is wearing the mask.

![Graph 1](/graph%201.png "Graph 1")
![Graph 2](/graph%202.png "Graph 2")

## 4. Demo

![Demo gif](/demo.gif "Demo gif.")

## 5. Conclusion
This project presented a convolutional neural network implementation used for live detection of masks on the face. The applied methodology has shown significant results with an average validation accuracy of 91.86% and an average training accuracy of 98.90%. As further work, the application could be optimized by providing a wider variety of dataset to train upon.
