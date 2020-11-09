---
title: 'Building & Visualizing your first ConvNet'
date: 2017-12-30
layout: post
tags:
  - CNN
  - Deep Learning
  - Convolutional Neural Networks
---

In the last few years alone a special kind of Neural Network known as the Convolutional Neural Network (ConvNet) has gained a lot of popularity, and for all the right reasons. ConvNets have a unique property of retaining *translational invariance*. In elaborative terms, they exploit spatially-local correlation by enforcing a local connectivity pattern between neurons of adjacent layers.

<img style="float: center;" src="https://raw.githubusercontent.com/sominwadhwa/sominwadhwa.github.io/master/images/bp2/translational_invariance.jpg">

To know more about ConvNets or Convolutions in general, you can read about them on [Christopher Olah's](http://colah.github.io/about.html) blog here --
1. [ConvNets: A Modular Perspective](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/)
2. [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)

My objective of writing this article is to get you upto speed on training deep learning models in the cloud without the hassles of setting up a VM, AWS instance or anything of that sort. After following through this article you'd be able to design your own classification task with lots of images & train your own deep learning models. All you'll need to implement the entire thing is a some knowledge of [Python](https://www.stavros.io/tutorials/python/), [GitHub](http://rogerdudler.github.io/git-guide/) & basics of [Keras](https://keras.io/) -- the quintessential DL starter. The data & implementation used here is inspired from [this post](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) on the official [Keras blog](https://blog.keras.io/).


## Setup

In any binary classification task our primary requirement is the data itself, more so, a dataset segregated into 2 classes. For the purpose, I'm sticking to a simple dataset -- [DogsVsCats](https://www.kaggle.com/c/dogs-vs-cats) from Kaggle. It is one of the most rudimentary classification task where we classify whether an image contains a dog or a cat. For simplicity, I've bundled 1000 images of Dogs & Cats each and created a directory with the following structure.

{% highlight python %}
.
.
├── train
	├── cats
	├── dogs
└── val
	├── cats
	├── dogs
└── test
.
.
{% endhighlight %}

The dataset can be accessed on FloydHub at [sominw/datasets/dogsvscats/1](https://www.floydhub.com/sominw/datasets/dogsvscats/1)

<img style="float: center;" src="https://raw.githubusercontent.com/sominwadhwa/sominwadhwa.github.io/master/images/bp2/cats-dogs.jpg">

If you haven't already setup an account on FloydHub, you can do so by using the [FloydHub QuickStart Documentation](http://docs.floydhub.com/getstarted/quick_start/). It's incredibly simple & if you're stuck at any point, they provide intercom chat support.


### Cloning the Git Repo

I've already prepared a starter code to use the above mentioned dataset that'll allow you to tinker around with the model we're building and make you understand the basics of training models with large datasets on Floyd. Navigate to a directory of your choice and enter the following --

{% highlight lineanchors %}
$ git clone https://github.com/sominwadhwa/DogsVsCats-Floyd.git
{% endhighlight %}

### Creating a project on Floyd

Once you're through with cloning the directory, it's time to initialise the project on Floyd.
1. Create & name your project under your account on FloydHub.com.
2. Locally, within the terminal, head over to the git repo (DogsVsCats-Floyd) managing the source code.

{% highlight lineanchors %}
$ floyd init [project-name]
{% endhighlight %}


## Code

### Pre-requisites

{% highlight python lineanchors%}
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Activation, Flatten, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
import h5py
from keras import backend as K
import numpy as np
{% endhighlight %}

* **NumPy**: NumPy is a scientific computing package in Python. In context of Machine Learning, it is primarily used to manipulate N-Dimensional Arrays & some linear algebra & random number capabilities. ([Documentation](https://docs.scipy.org/doc/numpy/))
* **Keras**: Keras is a high level neural networks API used for rapid prototyping. We'll be running it on top of TensorFlow, an open source library for numerical computation using data flow graphs. ([Documentation](https://keras.io/getting-started/sequential-model-guide/))
* **h5py**: Used simultaneously with NumPy to store huge amounts of numerical data in HDF5 binary data format. ([Documentation](http://docs.h5py.org/en/latest/quick.html#quick))
* **Tensorboard**: Tool used to visualize the static compute graph created by TensordFlow, plot quantitative metrics during the execution, and show additional information about it. ([Concise Tutorial](http://ischlag.github.io/2016/06/04/how-to-use-tensorboard/))

{% highlight python lineanchors %}
width, height = 150, 150
training_path = "/input/train"
val_path = "/input/val"
n_train = 2000
n_val = 400
epochs = 100
batch_size = 32
{% endhighlight %}

The above snippet defines the training & validation paths. `/input` is the default mount point of any directory (root) uploaded as 'data' on Floyd. The dataset used here is a publically accessible one.

### Model Architecture

{% highlight python lineanchors %}
model = Sequential()
model.add(Conv2D(32,(3,3), input_shape= input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
{% endhighlight %}


Keras makes it incredibly simple to sequentially stack fully configurable modules of neural layers, cost functions, optimizers, activation functions & regularization schemes over one another. For this demonstration, we've stacked three 2D ConvNet layers (1 Input, 2 Hidden) with [ReLu](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions) activation. To control overfitting, there's a 40% [dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) before the final activation in the last layer of the network along with MaxPooling layers. For the loss function, since this is a standard binary classification problem, `binary_crossentropy` is a standard choice. To read & learn more about Cross-Entropy loss, you can checkout [this article](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/) by Rob DiPietro.

[*Pooling*](http://cs231n.github.io/convolutional-networks/#pool): One indispensable part of a ConvNet is the Pooling Layer. It serves two primary purposes. By progressively reducing the spatial size of the representation, it retains 'translational invariance' in the network and by virtue of that it also reduces the amount of parameters and computation in the network, hence also controlling overfitting. Pooling is often applied with filters of size 2x2 with a stride of 2 at every depth slice. A pooling layer of size 2x2 with stride of 2 shrinks the input image to 1/4 of its original size.

<img style="float: center;" src="https://raw.githubusercontent.com/sominwadhwa/sominwadhwa.github.io/master/images/bp2/maxpool.jpeg">

### Data Preparation

Since we're using very little data (1k training examples per class), we try to augment these examples by a number of different image transformations using `ImageGenerator` class in Keras.  

{% highlight python lineanchors %}
train_data = ImageDataGenerator(
        rescale= 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train = train_data.flow_from_directory(
        training_path,
        class_mode='binary',
        batch_size=batch_size,
        target_size=(width,height))

{% endhighlight %}

So with a single image, we can generate a lot more belonging to the same class, containing the same object but in a slightly different form.

<img style="float: center;" src="https://raw.githubusercontent.com/sominwadhwa/sominwadhwa.github.io/master/images/bp2/ImageGenerator.jpg">

### TensorBoard

Tensorboard is a visualization tool provided with TensorFlow that allows us to visualize TensorFlow compute graphs among other things.

{% highlight python lineanchors %}
tensorboard = TensorBoard(log_dir='/output/Graph', histogram_freq=0, write_graph=True, write_images=True)
{% endhighlight %}

Keras provides [callbacks](https://keras.io/callbacks/) to implement TensorBoard among other procedures to keep a check on the internal states & statistics of the model during training. More so, FloydHub provides exclusive support for TensorBoard inclusion. For instance, the above snippet stores the TensorBoard logs in a directory `/Graph` & generates the graph in real time.

To know more about TensorBoard functionality & its usage head over to the official [documentation](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

### Training

So now that we've thoroghly dissected the code, it's finally time to train this network on the cloud. To run this job on Floyd, simply run the following in your terminal (after navigating to the project directory)

{% highlight lineanchors %}
floyd run --data sominw/datasets/dogsvscats/1:input --gpu --tensorboard "python very_little_data.py --logdir /output/Graph"
{% endhighlight %}

<img style="float: center;" src="https://raw.githubusercontent.com/sominwadhwa/sominwadhwa.github.io/master/images/bp2/train_ex.png">

1. `--logdir` flag provides a directory for storing the tensorboard logs.
2. `--gpu` (optional) indicates that you wish to use the GPU compute.
3. `--tensorboard` indicates the usage of TenorBoard.

Upon indicating that you're using **TensorBoard** (while executing the job), FloydHub provides a direct link to access the TenorBoard.

<img style="float: center;" src="https://raw.githubusercontent.com/sominwadhwa/sominwadhwa.github.io/master/images/bp2/TB_init.png">

To know more about TensorBoard's support on Floyd, you can checkout [this article](http://blog.floydhub.com/tensorboard-on-floydhub) by Naren Thiagarajan.

### Outputs

Keras lets you store multi dimensional numerical matrices in the form of weights in HDF5 Binary data format.

{% highlight python lineanchors %}
model.save_weights('/output/very_little_weights.hdf5')
{% endhighlight %}

The snippet above stores your generated weight file, at the end of training, to the `/Output` directory. And that's it! You've finally trained & visualized your first scalable ConvNet.

<img style="float: center;" src="https://raw.githubusercontent.com/sominwadhwa/sominwadhwa.github.io/master/images/bp2/result_convnet.png">

I'll encourage you to try out your own variants of ConvNets by editing the source code. In fact, you can refer to [this article](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) & build an even more powerful ConvNet by using pre-trained [VGG](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) weights.

If you'd like to read more about running instances on Floyd, using Datasets & running jobs with external dependencies, read my previous article on FloydHub's Blog: [Getting Started with Deep Learning on FloydHub](http://blog.floydhub.com/getting-started-with-deep-learning-on-floydhub)

## References

1. [Building powerful image classification models using very little data - Keras Official Blog](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
2. [ConvNets: A modular approach](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/)
3. [Emil Walner: My first weekend of Deep Learning](http://blog.floydhub.com/my-first-weekend-of-deep-learning)
4. [Intro to Cross Entropy Loss](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/)

------
