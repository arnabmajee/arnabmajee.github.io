---
title: 'Deep Learning on Floydhub'
date: 2017-08-29
layout: post
tags:
  - deep learning
  - neural networks
  - floydhub
---

This post is aimed at helping new users (especially the ones who are starting out & cannot afford [Andrej Karpathy's gig](https://twitter.com/karpathy/status/648256662554341377)) setup an on-the-go deep learning
solution for their small to medium sized projects. I'll be covering the following -
1. Getting started with your first DL project on Floyd.
2. Handling & usage of datasets.
3. Installation of external dependencies (if any) while running job instances on Floyd.

It all started a couple of days ago when I was scouting for a cheap CUDA supported
GPU solution for my college project (following soon) based on [VQA](http://visualqa.org/) (because GPUs aren't cheap here and college, well let's just say they're resourceful in their own subtle way).
I was well aware of the intricacies and cost implications of AWS (among others like Google Cloud, Azure and Bluemix) and was a bit hesitant to squander all my student credit in testing of rudimentary MLPs.

Recently, I heard about Floyd, a startup that is attempting to change that landscape. They claimed to be the `Heroku` of deep learning and, most importantly, had some affordable GPU options. More about it, [here](https://medium.com/@jrodthoughts/floyd-and-the-deep-learning-cloud-market-1ece81f717c7).

![floyd-home](https://raw.githubusercontent.com/sominwadhwa/sominwadhwa.github.io/master/images/bp1/floyd_img.png)

After going through their "extensive" documentation (which is thorough enough tbh) it took me around 31 (yes!) attempts to finally get going, which included understanding the directory & instance structure Floyd provides for a workable DL environment. So here I am, couple of days in, and absolutely enthralled by FloydHub giving my 2 cents to set it up and get going efficiently.

## Initial Setup

Initial setup is fairly simple and well elaborated in their documentation. It involves a simple sign up & installation of `floyd-cli` via `pip`. It's a standard process well explained [here](https://medium.com/@margaretmz/get-started-with-floydhub-82cfe6735795) as well.

## Project Initialization

After having created an account & setting up the `floyd-cli`, it is essential to understand how Floyd manages your data & source code separately (part of the reason it works wonderfully across various fronts while designing a deep learning pipeline).

1. On floydhub.com, create & name your new project following the instructions.
2. Locally, within the terminal, head over to the git repo (project directory) managing the source code. (Remember to keep your data- like the [VGG](https://keras.io/applications/#vgg16) weights etc in a separate directory- we'll get there in a bit).

{% highlight lineanchors %}
$ floyd init [project-name]
{% endhighlight %}

Once that's settled, you can try creating a test file to see if everything works. Nothing extensive, a simple sequential model with Keras should suffice. Here's one -

{% highlight python lineanchors %}
# test.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)

{% endhighlight %}

And `floyd run --gpu "python test.py"` should do the trick. You can check the status of this job via `floyd log` or in your browser. One subtle point to see here is that floyd will by default execute your source in a `tensorflow` based env. Should you use anything else (PyTorch/theano/etc), you have to specify that using the `--env` flag.

**Note**: Running the above script with `--gpu` flag will consume the GPU time allocated against your account on Floyd. In case you're just testing that everything is setup & works fine, you can avoid using the `--gpu` flag, allocating your instance for CPU compute.

## Managing Datasets

Since every time you shoot a `floyd run`, it synchronizes your code (the entire directory), it makes no absolute sense to store the training/testing/validation data (basically anything that isn't going to be altered for a very long time -- or possibly never) in a directory which is practically uploaded that frequently (Remember- Floyd does some sort of version control as well).

### Uploading Data

Now, say you're working with [MSCOCO](http://cocodataset.org/) and you've obtained the [VGG-net](https://keras.io/applications/#vgg16) weights - vectors of length 4096 once (that's what I was stuck with so..). And you being a nice chap, avoiding it to get into your `git` commit history, store it far away in some directory called `coco`. Put simply, now `coco` has all your data files (csv, mat, xlsx etc). Navigate to it, and do the following in order-

{% highlight lineanchors %}

$ floyd data init vgg-coco

{% endhighlight %}

{% highlight lineanchors %}
Data source "vgg-coco" initialized in current directory

    You can now upload your data to Floyd by:
        floyd data upload
{% endhighlight %}

{% highlight lineanchors %}

$ floyd data upload

{% endhighlight %}

{% highlight lineanchors %}

Compressing data...
Making create request to server...
Initializing upload...
Uploading compressed data. Total upload size: 729.0MiB
764440553/764440[================================] 764440553/764440553 - 00:39:13
Removing compressed data...
Upload finished.
Waiting for server to unpack data.
You can exit at any time and come back to check the status with:
	floyd data upload -r
Waiting for unpack.....

NAME
--------------------------
sominw/datasets/vgg-coco/1

{% endhighlight %}

If everything went well so far, you should have something like this appear in the `datasets` section of your profile --

![dataset1](https://raw.githubusercontent.com/sominwadhwa/sominwadhwa.github.io/master/images/bp1/dataset.png)

### Using Datasets

Floyd datasets are designed in a way that you can practically associate them with any project/script that you intend to run (just how you'd do on a local machine). All you have to do is **mount** them while creating your instance for the job using the `--data` flag. According to floyd's official documentation, to mount a specific version of a dataset, you need to specify its full name and the mount point. The syntax is `--data <data_name>:<mount_point>`.

The pivotal thing here is the mount point, which is essentially the name of the directory under which the data is available (however, it cannot contain subdirectories). By default, or if the entire data is dumped into the root of the folder which was uploaded as data, the mount point is `/input`. So in our case, the following will make the data available to the script under execution (`test.py`) --

{% highlight lineanchors %}
$ Somins-MacBook-Pro:VQAMD-Floyd sominwadhwa$ floyd run --data sominw/datasets/vgg-coco/1:input "python test.py"
{% endhighlight %}

Generic format for mounting data:

`--data <username>/datasets/<name_of_dataset>/<version-number>:<mount-point>`.

In order to access anything that was uploaded (present in our `coco` directory that was initialized earlier as data directory) within a script:

{% highlight lineanchors python %}
file_path = "/input/<filename>.csv"
{% endhighlight %}

### Navigating within the project directory

To navigate within the execution directory, floyd treats `/code` as `/root`. So say if you have a directory structure of the sort -

{% highlight python %}
.
├── test.py
└── temp
    ├── file.txt
{% endhighlight %}

`file.txt` can be accessed from within `test.py` during execution by specifying its path as --

{% highlight python %}
f = open("/code/temp/file.txt","rb").read()
{% endhighlight %}

### Storing Outputs

Much like most of the workflow, a separate `/output` is present within the directory structure Floyd provides, which is used to store outputs. It can store output logs as well --

* `floyd run "python helloworld.py > /output/my-output-file.txt"`
* `model.save_weights("/output/MLP" + "_epoch_{:02d}.hdf5".format(k))`

![output](https://raw.githubusercontent.com/sominwadhwa/sominwadhwa.github.io/master/images/bp1/output.png)

## Floyd Instances & Dependencies

Floyd comes with most core DL dependencies pre installed -- `numpy`,`scipy`,`SpaCy` etc. If in case, anything is additionally required, it can be specified within a file named `floyd_requirements.txt` present in the directory from where the instance of the job is launched. Some of the other detailed instructions regarding installing external dependencies is given in Floyd's official [documentation](http://docs.floydhub.com/guides/jobs/installing_dependencies/).

However, this one, in particular, got me stuck. I was trying to load `SpaCy` language models for English to access WordVectors but Floyd environments only has the binaries. You need to install language model for the language you're using yourself. The process is pretty straightforward **except** one small subtlety -- **Floyd Instances**.

Every time you run `floyd run .. ` command you get a **new** instance. What this means is that every time you execute your script, dependencies are to be reinstalled. For instance, if I run --  

`floyd run "python -m spacy download en"` & `floyd run "python test.py"`

It won't be of any use. Instead, we need to run both, the dependency installation and the script in a single instance --

`floyd run "python -m spacy download en && python test.py"`.

## Try it!

Eventually, if everything goes accordingly, you'd be looking at spinning a NVIDIA Tesla K80 Machine Learning Accelerator & watching it slice through image/text processing tasks like butter! ❤️

![output](https://raw.githubusercontent.com/sominwadhwa/sominwadhwa.github.io/master/images/bp1/output2.png)

Tried running your first instance on Floyd yet? Or ran into some issues? Either way, let me know your thoughts or just the time it took you to get going on Floyd in the comments section below!

## Resources

In case you wish to learn more about Floyd or simply about training in the cloud, checkout these links below:
1. [https://medium.com/simple-ai/power-cut-after-5-days-running-deep-learning-code-cda2317aad55](https://medium.com/simple-ai/power-cut-after-5-days-running-deep-learning-code-cda2317aad55)
2. [http://forums.fast.ai/t/floyd-alternative-to-aws-p2-instance/1394](http://forums.fast.ai/t/floyd-alternative-to-aws-p2-instance/1394)
3. [https://www.reddit.com/r/MachineLearning/comments/611mhs/p_floyd_zero_setup_deep_learning/](https://www.reddit.com/r/MachineLearning/comments/611mhs/p_floyd_zero_setup_deep_learning/)

------
