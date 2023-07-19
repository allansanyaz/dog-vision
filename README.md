* To use
```
conda env create -f environment.yml
conda activate myenv
python dog_breed_identifier.py -i <name-of-image-file>
```
For help:
```
python dog_breed_identifier.py -h
```

# dog-vision

# 🐶 End-to-end Multi-class Dog Breed Classification

This notebook builds an end-to-end multiclass classifier using Tensorflow 2.0 and Tensorflow Hub

## 1. Problem

Identifying the breed of adog given and image of a dog.

When I'm sitting at a cafe and I take a photo of a dog, I wantto know what breed of dog it is.

## 2. Data

The data we're using is from Kaggles dog breed idenitification competition.

"https://www.kaggle.com/c/dog-breed-identification"

## 3. Evaluation


The evaulation is a file with prediction probabilities for each dog breed with each test image.

"https://www.kaggle.com/c/dog-breed-identification/overview/evaluation"

## 4. Features

Some information about the data:
* We're dealing with images (unstructured data) so it's probably best we use deep learning/tranfer learning.
* There are 120 breeds of dogs (this means that there are 120 different classes).
* There are around 10,000+ images in the training set (these images have labels)
* There are around 10,000+ images in the test set (these images have no labels becuase we'll want to predict them)


## Get our workspace ready

* Import TensorFlow 2.x ✔
* Import TensorFlow Hub ✔
* Make sure we're using a GPU ✔

```
# Import neccessary tools
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.python.client import device_lib

print("TF version:", tf.__version__)
print("TF hub version:", hub.__version__)
# print("Checking to see if the GPU is here: \n", device_lib.list_local_devices())
# Check for GPU availability
print("GPU","available (YESSSSS!!!!!)" if tf.config.list_physical_devices("GPU") else print("Not there :-("))

```
** Should give the below output
```
TF version: 2.4.1
TF hub version: 0.11.0
GPU available (YESSSSS!!!!!)
```

## Getting our data ready (turning it into Tensors)

With all machine learning models, our data has to be in numerical format. So thats what we'll be doing first. Turning our images into Tensors (numerical representations)

Lets start by accessing our data and checking out the labels

```
# Checkout the labels of our data
import pandas as pd

labels_csv = pd.read_csv("dog-breed-identification{0}labels.csv".format(os.sep))
print(labels_csv.describe())
print(labels_csv.head(5))
```
** Will give the output
```
                                      id               breed
count                              10222               10222
unique                             10222                 120
top     0490b67cb414d527d6c21052b1e3b5dd  scottish_deerhound
freq                                   1                 126
                                 id             breed
0  000bec180eb18c7604dcecc8fe0dba07       boston_bull
1  001513dfcb2ffafc82cccf4d8bbaba97             dingo
2  001cdf01b096e06d78e9e5112d419397          pekinese
3  00214f311d5d2247d5dfe4fe24b2303d          bluetick
4  0021f9ceb3235effd7fcde7f7538ed62  golden_retriever
```
Chek the number of images for each breed
```
# How many images are there of each breed?
labels_csv["breed"].value_counts().plot.bar(figsize=(20,10))
```

** This will show
![image](https://github.com/allansanyaz/dog-vision/assets/87567534/11989135-8e33-4e42-b8b1-61403253ad5f)

```
labels_csv["breed"].value_counts().median()
path = "dog-breed-identification{0}".format(os.sep)
dogvision_path = "".format(os.sep)
```

##### With regards to the minimal number of images, google recommends a minimum of 10 image per class. 100 is great, but 10 is a great start.
Lets view an image
```
from IPython.display import display, Image
# Programming/Machine_learning/projects_unstructured/dog_vision/dog-breed-identification/train/000bec180eb18c7604dcecc8fe0dba07.jpg
Image(f"{path}train{os.sep}000bec180eb18c7604dcecc8fe0dba07.jpg")
```
![output](https://github.com/allansanyaz/dog-vision/assets/87567534/7b06d1ea-f45d-460a-a256-8e30b42df5c6)

```
labels_csv.head()
```
** WIll show
```
                                id	breed
0	000bec180eb18c7604dcecc8fe0dba07	boston_bull
1	001513dfcb2ffafc82cccf4d8bbaba97	dingo
2	001cdf01b096e06d78e9e5112d419397	pekinese
3	00214f311d5d2247d5dfe4fe24b2303d	bluetick
4	0021f9ceb3235effd7fcde7f7538ed62	golden_retriever
```

### Getting images and their labels 
Let's get a list of images and their path names

```
# Adding full path name to a list for better access and comprehension
filenames = [f"{path}train{os.sep}{id}.jpg" for id in labels_csv["id"]]

# Check to see whether the number of filenames matches the actual number of image files
print("File numbers are","Equal" if len(filenames) == len(os.listdir(f"{path}train")) else "Not Equal")
```
** Will show
```
File numbers are Equal
```

One more check
```
Image(filenames[9000])
```
![output_1](https://github.com/allansanyaz/dog-vision/assets/87567534/2829bf6e-9353-4b04-aa91-cac40bc07f4a)

```
labels_csv["breed"][9000]
```

** The breed is:
```
'tibetan_mastiff'
```

Since we've now got our training image filepaths in a list, lets prepare our labels 
```
import numpy as np
labels = labels_csv["breed"].to_numpy()
# labels = np.array(labels) # does the same thing as above
labels
```
** Output:
```
array(['boston_bull', 'dingo', 'pekinese', ..., 'airedale',
       'miniature_pinscher', 'chesapeake_bay_retriever'], dtype=object)
```

```
len(labels)
```

```
10222
```

See if numpy of labels matches the numbers of filenames
```
print("Numbers are:","Equal" if(len(labels) == len(filenames)) else "Not Equal")
```
```
Numbers are: Equal
```

```
unique_breeds = np.unique(labels)
unique_breeds
len(unique_breeds)
```

Turn a single label into an array of booleans
```
print(labels[0])
labels[0] == unique_breeds
```
** Output:
```
boston_bull
array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False,  True, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False])
```

Turn every label into a boolean array by comparing  each label in the full list to unique breeds to ge True False values
```
boolean_labels = [label == unique_breeds for label in labels]
boolean_labels[:2]
```
** Output:
```
[array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False,  True, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False]),
 array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False,  True, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False])]
```
** Output:
```
len(boolean_labels)
```
** Output:
```
10222
```

Example: Turning boolean array into integers
```
print(labels[0]) # Original label
print(np.where(unique_breeds == labels[0])) # index where label occurs
print(np.argmax(boolean_labels[0])) # index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # they'll be a 1 where there truth occurs
```
** Output:
```
boston_bull
(array([19], dtype=int64),)
19
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0]
```

```
boolean_labels[:2]
```
** Output:
```
[array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False,  True, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False]),
 array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False,  True, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False])]
```

Take a look at the file names
```
filenames[:10]
```
** Output:
```
['dog-breed-identification\\train\\000bec180eb18c7604dcecc8fe0dba07.jpg',
 'dog-breed-identification\\train\\001513dfcb2ffafc82cccf4d8bbaba97.jpg',
 'dog-breed-identification\\train\\001cdf01b096e06d78e9e5112d419397.jpg',
 'dog-breed-identification\\train\\00214f311d5d2247d5dfe4fe24b2303d.jpg',
 'dog-breed-identification\\train\\0021f9ceb3235effd7fcde7f7538ed62.jpg',
 'dog-breed-identification\\train\\002211c81b498ef88e1b40b9abf84e1d.jpg',
 'dog-breed-identification\\train\\00290d3e1fdd27226ba27a8ce248ce85.jpg',
 'dog-breed-identification\\train\\002a283a315af96eaea0e28e7163b21b.jpg',
 'dog-breed-identification\\train\\003df8b8a8b05244b1d920bb6cf451f9.jpg',
 'dog-breed-identification\\train\\0042188c895a2f14ef64a918ed9c7b64.jpg']
```

### Creating our own validation set
Since the dataset from Kaggle doesnt some with a validation set we are going to create our own.
```
# Setup X & y variables
X = filenames
y = boolean_labels
```

#### We're going to start off with experimenting with 1000 images and incease as needed
Set number of images to use for experimenting
```
NUM_IMAGES = 5000 #@param {type:"slider", min:1000, max:10000, step:1000}
```

Lets split the data into train and validation sets
```
from sklearn.model_selection import train_test_split

# Split them into training and validation of total size NUM_IMAGES
X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES],
                                                  y[:NUM_IMAGES],
                                                  test_size=0.2,
                                                  random_state=42)

len(X_train), len(X_val), len(y_train), len(y_val)
```
** Output:
```
(4000, 1000, 4000, 1000)
```

Lets have a look at the training data
```
X_train[:2], y_train[:2]
```
** Output:
```
Output exceeds the size limit. Open the full output data in a text editor
(['dog-breed-identification\\train\\693f4cd00978df07e1283d3da4d02e0c.jpg',
  'dog-breed-identification\\train\\7521421e092333c78d6b9dc2e189e659.jpg'],
 [array([False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False,  True, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False]),
  array([False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
...
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False,  True, False, False, False, False, False,
         False, False, False])])
```

### Preprocessing images (turning images into Tensors)

To preprocess our images intoTensors we're going to write a function which does a few things:
1. Take an image filepath input
2. Use TensorFlow to read the file and save it to a variable `image`
3. Turn our `image` (a jpg) into Tensors
4. Resize the image to be a shape of (224, 224)
5. Return the modified image

Before we do, let's see what importing an image looks like

#### Convert an image to a numpy array
```
from matplotlib.pyplot import imread
image = imread(filenames[9000])
image.shape
```

```
(600, 610, 3)
```

```
image[:2]
```
** Output:
```
array([[[124, 129,  88],
        [150, 155, 114],
        [122, 127,  86],
        ...,
        [ 80,  85,  55],
        [ 66,  73,  40],
        [ 66,  73,  40]],

       [[144, 149, 108],
        [153, 158, 117],
        [121, 126,  85],
        ...,
        [ 58,  63,  33],
        [ 65,  72,  39],
        [ 65,  72,  39]]], dtype=uint8)
```

```
tf.constant(image)[:2]
```
** Output:
```
<tf.Tensor: shape=(2, 610, 3), dtype=uint8, numpy=
array([[[124, 129,  88],
        [150, 155, 114],
        [122, 127,  86],
        ...,
        [ 80,  85,  55],
        [ 66,  73,  40],
        [ 66,  73,  40]],

       [[144, 149, 108],
        [153, 158, 117],
        [121, 126,  85],
        ...,
        [ 58,  63,  33],
        [ 65,  72,  39],
        [ 65,  72,  39]]], dtype=uint8)>
```

Now we've see what an image looks like as a Tensor, lets make a function to preprocess them.

1. Take an image filepath input
2. Use TensorFlow to read the file and save it to a variable `image`
3. Turn our `image` (a jpg) into Tensors
4. Normalise our image (convert colour channel values from 0-255 to 0-1
5. Resize the image to be a shape of (224, 224)
6. Return the modified image

## ☘ It is essential to feed data in the same format and shape as the model was trained on. ☘

```
# Define image size,
IMG_SIZE = 224

# Create a function for preprocessing images
def process_image(image_path, img_size=IMG_SIZE):
    '''
    Takes an image file path and turns the image into a tensor
    '''
    # Read in an image file
    image = tf.io.read_file(image_path)
    # Turn the jpg image into numeric tensor with three colour channels RGB
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert the colour channel values from 0 to 255 to 0 to 1 values
    # Normlising the values makes tf perform the functions easier
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Resize the image to our desired value (224, 224)
    image = tf.image.resize(image, [img_size, img_size])

    return image
```

TensorFlow likes to process information in batches
* Yann Lecun batch size (don't use batch sizes larger than 32)
* Jeremy howard batch size

```
tensor = tf.io.read_file(filenames[52])
tensor = tf.image.decode_jpeg(tensor, channels=3) # similar to the matplotlib imread option
tf.image.convert_image_dtype(tensor, dtype=tf.float32)
```
** Output:
```
Output exceeds the size limit. Open the full output data in a text editor
<tf.Tensor: shape=(288, 288, 3), dtype=float32, numpy=
array([[[0.16862746, 0.12156864, 0.07450981],
        [0.18431373, 0.13725491, 0.09019608],
        [0.19215688, 0.14509805, 0.09803922],
        ...,
        [0.28627452, 0.16862746, 0.1254902 ],
        [0.26666668, 0.14901961, 0.10980393],
        [0.24705884, 0.12941177, 0.09019608]],

       [[0.16862746, 0.12156864, 0.07450981],
        [0.18039216, 0.13333334, 0.08627451],
        [0.18431373, 0.13725491, 0.09019608],
        ...,
        [0.28235295, 0.16470589, 0.12156864],
        [0.26666668, 0.14901961, 0.10980393],
        [0.2509804 , 0.13333334, 0.09411766]],

       [[0.1764706 , 0.12941177, 0.08235294],
        [0.18431373, 0.13725491, 0.09019608],
        [0.18039216, 0.13333334, 0.08627451],
        ...,
        [0.2784314 , 0.16078432, 0.11764707],
        [0.26666668, 0.14901961, 0.10588236],
        [0.25882354, 0.14117648, 0.09803922]],
...
        [0.12941177, 0.12941177, 0.13725491],
        ...,
        [0.42352945, 0.2627451 , 0.13725491],
        [0.41176474, 0.25882354, 0.12941177],
        [0.40784317, 0.25490198, 0.1254902 ]]], dtype=float32)>
```

## Turning data into batches

Why turn our data into batches?

Calculate in loops or bacthes...

Let's say you're trying to process 10,000+ images in one go... they all might not fit into memory.

So thats why we do about 32 (this is batch size) images at a time (you can manually adjust batch size if need) 

In order to use TensorFlow effectively we need our data in a form of tensor tupples that look like this:
`(image, label)`

```
def get_image_label(image_path, label):

    '''
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image,label)
    '''

    image = process_image(image_path)

    return image, label
```

👀 Same steps are taken for text as well. Everything needs to be in a tensor. The preprocessing is however what remains different in that we have handled this for images so we'd need to do the same for text.

```
# Demo of the above
(process_image(X[13]), tf.constant(y[13]))
```
** Output:
```
Output exceeds the size limit. Open the full output data in a text editor
(<tf.Tensor: shape=(224, 224, 3), dtype=float32, numpy=
 array([[[0.47678575, 0.62188375, 0.30815828],
         [0.51782215, 0.6629202 , 0.34919468],
         [0.5327031 , 0.67780113, 0.36407563],
         ...,
         [0.56141454, 0.6717262 , 0.42188376],
         [0.49532533, 0.6122896 , 0.36130923],
         [0.48235297, 0.6       , 0.34901962]],
 
        [[0.486211  , 0.63130903, 0.31758353],
         [0.4844512 , 0.6295492 , 0.31582373],
         [0.4681175 , 0.61321557, 0.29949003],
         ...,
         [0.58139545, 0.6950981 , 0.43859532],
         [0.56051975, 0.6774841 , 0.42514732],
         [0.5563638 , 0.67768013, 0.41569197]],
 
        [[0.5086412 , 0.6537392 , 0.34776926],
         [0.48860604, 0.6337041 , 0.32773417],
         [0.4582992 , 0.60339725, 0.29742733],
         ...,
         [0.5842827 , 0.6980082 , 0.43645748],
         [0.55202645, 0.6721934 , 0.40956438],
         [0.5452294 , 0.66679543, 0.40040523]],
 
...
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False])>)
```

No we've got a way to turn our data into tuples or Tensors in the form `(image, label)`,
let's make a function to turn all of our data X and y into batches

```
# Define the batch size thats a good start
BATCH_SIZE = 32

# Create a function to turn data into batches. Each for the training, validation, and test
def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    '''
    Creates bacthes of data out of the image (X) and label (y) pairs.
    Shuffles the data  if its training data but doesn't shuffle if its validation data.
    Also accepts test data as input (no labels)
    '''
    X = tf.constant(X)
    y = tf.constant(y) if(not y == None) else None
    # If data is a test dataset, we probably don't have label
    if(test_data):
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((X)) # only filepaths (no labels)
        data_batch = data.map(process_image).batch(BATCH_SIZE) # the slice data set is what is going to the fucntion with map so the data becomes the parameter

        return data_batch

    # If the data is a validation (valid_data) dataset, we don't need to shuffle it??
    # Why?? because the training data maybe in some sequence but its not important for the validation data
    elif(valid_data):
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((X, # filepaths
                                                   y)) # labels
        data = data.map(get_image_label)

        data_batch = data.batch(BATCH_SIZE)

        return data_batch

    else:
        print("Creating training data bacthes")
        # X = tf.random.shuffle(X, seed=42) 
        # y = tf.random.shuffle(y, seed=42)

        data = tf.data.Dataset.from_tensor_slices((X, 
                                                   y))
        # Shuffling pathnames and labels before mapping is faster than shuffling images
        data = data.shuffle(buffer_size=len(X))

        # Create (image, label) tuples (this also turns the image path into a preprocessed image) 
        data = data.map(get_image_label)

        # Turn the training data into batches
        data_batch = data.batch(BATCH_SIZE)

        return data_batch
```

```
# Creating training an and validation data batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)
```

```
Creating training data bacthes
Creating validation data batches...
```

Check out the different attributes of our data batches
```
train_data.element_spec, val_data.element_spec
```
** Output:
```
((TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None),
  TensorSpec(shape=(None, 120), dtype=tf.bool, name=None)),
 (TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None),
  TensorSpec(shape=(None, 120), dtype=tf.bool, name=None)))
```

### Visualing Data Batches

Our data is now in batches however these can be a little hard to undestand/comprehendlet's visualise them.

```
import matplotlib.pyplot as plt

# Create a function for viewing images in a data batch
def show_25_images(images, labels):
    '''
    Displays a plot of 25 images and their labels from a data batch.
    '''
    # Setup the figure
    plt.figure(figsize=(10,10))
    # Loop thtough 25 (for dislaying 25 images)
    for i in range(25):
        # Create subplots (5 rows, 5 columns)
        ax = plt.subplot(5, 5, i+1)
        # Display an image
        plt.imshow(images[i])
        # Add the image label as the title
        # Unlike elsewhere, labels here is the True, False matrices, so index of the maximum value
        plt.title(unique_breeds[np.argmax(labels[i])])
        # Turn the gridlines off
        plt.axis("off")
```

The train_data is a batch thus next allows us to remove the top item and the as iterator turns it into an iterator

```
train_images, train_labels = next(train_data.as_numpy_iterator())
```

```
# Now lets visualise the data in a training batch
train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images,train_labels)
```

![image](https://github.com/allansanyaz/dog-vision/assets/87567534/9a33e12f-ac21-434d-bb4d-f77e9b6eedb7)

```
val_images, val_labels = next(val_data.as_numpy_iterator())
show_25_images(val_images, val_labels)
```

![image](https://github.com/allansanyaz/dog-vision/assets/87567534/da1bf599-de10-451e-a1d0-376d3fc24c90)

## 🏗 Building a model

Before we build a model, there numerous ways to build a deep learning model but we will use transfer learning.

We need to define a few things:
* The input shape (our image shape, in the form of Tensors) to our model
* THe output shape (image labels, in the form of Tensors) of our model
* The URL of the model we want to use from TensorFlow Hub -:
"https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"

```
# Setup input shape to the model
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # batch, height, width, colour channels

# Setup output shape of our model
OUTPUT_SHAPE = len(unique_breeds)

# Setup model URL from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
```

```
Now we've got our inputs, outputs, and model ready to go. Lets put them together into a Keras deep learning model!

Knowing this, lets create a function which:
* Takes the input shape, output shape and the model we've chosen as parameters
* Defines the layers in the Keras model in a sequential fashion (do this first, then this, then that)
* Compiles the model (says it should be evaluated and improved)
* Builds the model (tells the model the input shape it'll be getting)
* Returns the model

All of these steps can be found here:
"https://www.tensorflow.org/guide/keras/sequential_model"
```

```
# Create a function which builds a Keras model
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    print("Building model with:", MODEL_URL)

    # Setup the model layers
    model = tf.keras.Sequential([hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
                                tf.keras.layers.Dense(units=output_shape,
                                activation="softmax") # layer 2 (output layer)
                                 ])
    
    # KerasLayer is the model we import and its respective features than we then reduce
    # Layers are those circles on the neural nets graph and we tell it that we dont need the 1280 of the algorithm we just need 120

    # Compile the model
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # loss measures how well the machine is learning. So it is trying to find the minimum loss. The higher the loss the worse the model is at learning predictions.
    # The higher you are up a hill at the descending championships the worse youre doing

    # Adam() lowers the loss function, thus he tells me how to get down the hill. The adam optimiser is a general one that performs well on hills

    model.build(input_shape)

    return model
```

```
model = create_model()
model.summary()
```
** Output:
```
Building model with: https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
keras_layer (KerasLayer)     (None, 1001)              5432713   
_________________________________________________________________
dense (Dense)                (None, 120)               120240    
=================================================================
Total params: 5,552,953
Trainable params: 120,240
Non-trainable params: 5,432,713
_________________________________________________________________
```

```
model
```
** Output:
```
<tensorflow.python.keras.engine.sequential.Sequential at 0x1caff6a32e0>
```

## Creating callbacks

Callbacks are helper functions a model can use during training to do things such as:
1. Save its progress
2. Check its progress
3. Stop training early if a model stops improving

We'll create two call backs one for TensorBoard which helps track our models progress and anothers for early stopping which prevents our model from training for too long which can result in over fitting.

## TensorBoard Callback

To setup a TensorBoard callback, we need to do 3 things:
1. Load the TensorBoard extension ✔
2. Create a TensorBoard callback which is able to save logs to a directory and pass it to our model's `fit` function ✔
3. Visualise our models training logs with the `%tensorboard` magic function (we'll do this while model training)


### Load Tensorboard notebook extension
```
%load_ext tensorboard
```

```
import datetime

# Create a function to build a TensorBoard callback
def create_tensorboard_callback():
    # Create a log directory for storing TensorBoard logs
    log_dir = os.path.join(f"{dogvision_path}logs",
                          # Make it so the logs get tracked whenever we run an experiment
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    return tf.keras.callbacks.TensorBoard(log_dir)
```

### Early stopping callback

Early stopping helps stop our model from overfitting by stopping training if a certain evaluation metric stops improving

https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping

```
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3)
```

## Training a model (on subset of data)

Our first model is only going to train on 1000 images to make sure everything is working
```
NUM_EPOCHS = 100 #@param {type:"slider", min:10, max:100, step:10}
```

```
# Check to make sue we're still running on aGPU
print("GPU","available (YES!)" if tf.config.list_physical_devices("GPU") else "not vailable")
```
** Output:
```
GPU available (YES!)
```

Let's create a function which trains a model.

* Create a model using `create_model()`
* Setup a TensorBoard callback using `create_tensorboard_callback()`
* Call the `fit()` function on our model passing it the training data, validation data, number of epochs to train for (NUM_EPOCHS) and the callbacks we'd like to use
* return the model

```
# Build a function to train and return a trained model
def train_model():
    '''
    Trains a given model and returns the trained version
    '''
    # Create a model
    model = create_model()

    # Create new TensorBoard session everytime we train a model
    tensorboard = create_tensorboard_callback()

    # Fit a model to the data passing it the callbacks we created
    model.fit(x=train_data,
              epochs=NUM_EPOCHS, # number of times to scan the image for patterns
              validation_data=val_data,
              validation_freq=1, # how many epochs to take to validate model. Validate every 1 epoch
              callbacks=[tensorboard, early_stopping])
    
    # Return the fitted model
    return model
```

Fit the model to the data
```
model = train_model()
```
** Output:
```
Building model with: https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
Epoch 1/100
125/125 [==============================] - 97s 699ms/step - loss: 3.4691 - accuracy: 0.2850 - val_loss: 0.9414 - val_accuracy: 0.7320
Epoch 2/100
125/125 [==============================] - 18s 147ms/step - loss: 0.4797 - accuracy: 0.8823 - val_loss: 0.7757 - val_accuracy: 0.7780
Epoch 3/100
125/125 [==============================] - 19s 148ms/step - loss: 0.2327 - accuracy: 0.9607 - val_loss: 0.7370 - val_accuracy: 0.7640
Epoch 4/100
125/125 [==============================] - 19s 148ms/step - loss: 0.1380 - accuracy: 0.9834 - val_loss: 0.7245 - val_accuracy: 0.7750
Epoch 5/100
125/125 [==============================] - 19s 147ms/step - loss: 0.0944 - accuracy: 0.9927 - val_loss: 0.7179 - val_accuracy: 0.7740
```

**Question?** It looks like our model is overfitting because it is performing way better on the training model than on the validation model
1. What are some ways to prevent model overfitting in deep learning neural networks

* Overfitting is a good thing, it shows that our model is learning

### Getting ready to read the log files

**Checking the TensorBoard logs**

The TensorBoard magic function (`%tensorboard`) will access the logs directory we created earlier and visualise its contents

```
# %load_ext tensorboard
%tensorboard --logdir logs
```
** Output:
```
ERROR: Timed out waiting for TensorBoard to start. It may still be running as pid 24540.
```

### Making and evaluating predictions using a trained model

```
# Make predictions on the validation data (not used to train on)
predictions = model.predict(val_data, verbose=1)
predictions # associated probaility of what our model thinks the image is
```
** Output:
```
32/32 [==============================] - 4s 112ms/step
array([[5.8006594e-06, 1.2682365e-07, 1.1809556e-07, ..., 4.6824980e-06,
        4.1858937e-05, 1.1738575e-06],
       [3.4373326e-05, 7.8948756e-07, 8.4183216e-07, ..., 1.3381808e-08,
        5.4497614e-06, 1.0502736e-05],
       [1.3964185e-04, 2.7022636e-06, 1.6485952e-04, ..., 3.4261622e-05,
        5.1791681e-04, 5.2394897e-02],
       ...,
       [5.4606726e-08, 1.5825403e-06, 1.4248210e-06, ..., 1.1652230e-06,
        2.2012035e-05, 8.3154413e-07],
       [1.6598568e-04, 6.3577795e-06, 4.5301766e-07, ..., 5.9566406e-05,
        1.8784084e-03, 2.8372479e-03],
       [1.7966317e-05, 1.0332699e-02, 9.7228540e-06, ..., 3.6041853e-05,
        3.1868476e-04, 6.9568225e-05]], dtype=float32)
```

```
predictions.shape
```
** Output:
```
(1000, 120)
```

```
len(y_val)
```
** Output:
```
1000
```

```
len(unique_breeds)
```
** Output:
```
120
```

```
predictions[0]
```
** Output:
```
Output exceeds the size limit. Open the full output data in a text editor
array([5.80065944e-06, 1.26823650e-07, 1.18095564e-07, 2.42230653e-05,
       3.67938701e-05, 2.34664284e-07, 1.37024099e-05, 2.04559370e-07,
       7.57192879e-07, 2.99110379e-05, 3.17746299e-06, 1.94958076e-07,
       2.68777455e-07, 1.16903735e-07, 1.53368046e-06, 3.41408645e-06,
       4.51870619e-07, 9.98567700e-01, 7.85200825e-07, 3.55182533e-06,
       1.42264116e-06, 1.46938964e-05, 1.05284147e-04, 1.19809420e-05,
       6.65217570e-08, 1.40916507e-06, 3.15637590e-04, 5.75219076e-07,
       4.40014844e-07, 3.97312806e-06, 1.04850422e-07, 1.02554885e-07,
       8.67500148e-07, 6.68408347e-07, 1.72092555e-07, 3.75617560e-06,
       2.00438285e-06, 3.73781717e-07, 1.37761518e-07, 2.29238026e-06,
       4.38266767e-07, 5.31545879e-07, 4.71065476e-07, 1.97706572e-06,
       1.97186779e-07, 4.26677161e-07, 1.23441187e-05, 2.04791172e-06,
       1.79235922e-05, 5.27229304e-07, 4.83604310e-07, 4.95981908e-07,
       1.17709833e-08, 2.85756414e-06, 5.81610005e-09, 1.45035088e-07,
       1.66449354e-07, 1.60158761e-05, 9.26011410e-08, 7.93545769e-05,
       1.90495456e-07, 4.43618440e-08, 5.41793497e-07, 1.88957404e-06,
       9.27780718e-07, 1.36729355e-07, 2.26034899e-06, 4.45287151e-06,
       1.70956118e-04, 1.55917969e-05, 2.34775314e-07, 1.04755472e-07,
       1.11247236e-05, 1.15559033e-06, 9.25431414e-06, 1.87446358e-06,
       2.76217179e-07, 7.54104531e-06, 1.46187389e-07, 8.12589569e-05,
       1.53814221e-06, 3.30786133e-05, 4.44005025e-07, 1.37214620e-05,
       6.41126405e-07, 1.55947859e-07, 5.53197481e-07, 5.49969286e-07,
       1.93833966e-06, 4.52646987e-07, 1.10563508e-06, 2.84741463e-06,
       5.63775984e-07, 1.12743578e-07, 4.19112389e-08, 2.15356590e-06,
       2.19164781e-06, 6.94287446e-05, 9.00731538e-06, 7.66968924e-08,
...
       3.62243486e-06, 7.47855822e-08, 1.56523529e-04, 5.23001177e-07,
       2.44960063e-07, 1.56207761e-07, 3.00155904e-07, 3.24081543e-06,
       1.43515408e-07, 1.44767412e-07, 2.26237930e-07, 6.77413766e-07,
       4.70935493e-06, 4.68249800e-06, 4.18589370e-05, 1.17385753e-06],
      dtype=float32)
```

```
np.sum(predictions[0])
```

```
1.0000002
```

First prediction
```
def dog_predictor(index=0):
    print(predictions[index])
    print(f"Max value (probability of prediction): {np.max(predictions[index])}")
    print(f"Sum: {np.sum(predictions[index])}")
    print(f"Max index: {np.argmax(predictions[index])}")
    print(f"Predicted label: {unique_breeds[np.argmax(predictions[index])]}")
    print(f"The dog is actually a: {unique_breeds[np.argmax(y_val[index])]}") # comparing to y_val allows us to check the identity of the dog. Another function involving unbatching is available furtehr below

dog_predictor(13)
```
** Output:
```
Output exceeds the size limit. Open the full output data in a text editor
[6.56242046e-05 2.80046515e-04 2.66500632e-04 2.18040550e-05
 1.33508902e-05 3.09700590e-06 3.19147366e-05 2.33745632e-05
 9.64768333e-06 9.02557417e-07 8.40484972e-06 9.19853410e-05
 5.07471304e-05 1.12138730e-06 4.89682097e-05 8.68303978e-05
 3.43129184e-04 5.55989300e-06 2.63616326e-04 4.67959580e-06
 1.61754841e-03 4.05393348e-06 1.82768505e-04 1.97225614e-04
 2.31460263e-06 1.30364815e-05 8.85571499e-05 1.03569648e-04
 2.43922204e-05 6.97523137e-06 3.49950365e-04 6.02363843e-06
 3.05290341e-05 4.78703529e-04 2.53978214e-04 4.44283360e-05
 3.29593749e-04 5.68833493e-04 1.80548504e-05 1.29442751e-05
 8.83492078e-07 6.79789764e-06 7.66310950e-06 4.03823471e-03
 4.24619975e-05 4.00781391e-05 1.98353425e-01 2.08491042e-06
 1.95973174e-04 1.11272921e-05 1.82027052e-05 2.25198237e-05
 1.23930822e-05 3.79444245e-05 6.65583551e-01 5.14854037e-05
 6.88518630e-05 6.58935533e-05 6.34834141e-05 3.84678773e-04
 1.51656823e-05 6.05982768e-06 1.34400988e-03 5.83374687e-03
 7.55404108e-05 1.49559837e-05 1.19301192e-04 9.57767224e-06
 2.24797841e-05 8.96380167e-04 1.20336119e-06 2.18514027e-03
 3.62346484e-03 3.14248581e-07 1.27212552e-04 5.12168917e-05
 9.15212695e-06 4.67975951e-05 5.66955598e-04 9.62586910e-07
 9.04849172e-03 3.87823966e-04 2.61164532e-05 3.74343981e-05
 5.81830591e-06 2.06971872e-05 8.79489016e-05 6.72152382e-05
 1.97472455e-05 7.02077741e-05 8.29658984e-06 3.01144678e-06
 5.98499810e-06 1.76138114e-04 4.94230590e-05 9.70279798e-02
 3.08573217e-04 2.02927928e-04 1.10146484e-05 5.11604885e-04
...
Sum: 1.0000001192092896
Max index: 54
Predicted label: groenendael
The dog is actually a: groenendael
```

```
unique_breeds[73]
```

```
'maltese_dog'
```

Having the above functionality is great but we want to be able to do it at scale.

And it would even be better to see the image the prediction the image is being made on!

**Note:** Prediction probailities are also know as confidence levels

```
# Turn probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
    '''
    Turns an array of predictiion probabilites into a label.
    '''
    return unique_breeds[np.argmax(prediction_probabilities)]

# Get a predicted label based on an array of prediction probabilities
pred_label = get_pred_label(predictions[198])
pred_label
```

```
'affenpinscher'
```

Now since our validation data is still in a batch dataset, we'll have to unbatchify it to make predictions on the validation images and then compare those predictions to the validation labels (truth labels)

```
images_tf = []
labels_tf = []

# Loop through unbatched data
for image, label in val_data.unbatch().as_numpy_iterator():
    images_tf.append(image)
    labels_tf.append(label)

images_tf[0], labels_tf[0]
```
** Output:
```
Output exceeds the size limit. Open the full output data in a text editor
(array([[[0.08661207, 0.09751292, 0.07815619],
         [0.06751302, 0.0918505 , 0.05747823],
         [0.0604112 , 0.09729353, 0.04384629],
         ...,
         [0.83837754, 0.8455904 , 0.79945946],
         [0.7929071 , 0.7970798 , 0.7295842 ],
         [0.8214812 , 0.826038  , 0.75354415]],
 
        [[0.07184874, 0.09509204, 0.08496422],
         [0.06569613, 0.096114  , 0.0656529 ],
         [0.04666601, 0.09603959, 0.03923428],
         ...,
         [0.74699265, 0.75564605, 0.6929895 ],
         [0.72310936, 0.73635286, 0.6507261 ],
         [0.75028026, 0.76419234, 0.67546123]],
 
        [[0.04185049, 0.07891282, 0.07907913],
         [0.03230042, 0.08288307, 0.05734802],
         [0.02757026, 0.09470031, 0.03129049],
         ...,
         [0.65601474, 0.67997307, 0.56799006],
         [0.6598779 , 0.6838362 , 0.57188773],
         [0.68698364, 0.710942  , 0.5989934 ]],
 
        ...,
...
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False]))
```

```
get_pred_label(labels_tf[0])
```

```
'border_terrier'
```

```
get_pred_label(predictions[0])
```

```
'border_terrier'
```

```
# Creating a function to unbachify a dataset
def unbatchify(image_label_batch):
    '''
    Takes a batched dataset (image, label) Tensors and Returns separate arrays of images and labels
    '''
    images_holder = []
    labels_holder = []

    for image, labels in image_label_batch.unbatch().as_numpy_iterator():
        images_holder.append(image)
        labels_holder.append(unique_breeds[np.argmax(labels)])

    return images_holder, labels_holder

val_images, val_labels = unbatchify(val_data)
# pred_images, pred_labels = unbatchify(predictions)
```

```
val_images[2], val_labels[2]
```
** Output:
```
Output exceeds the size limit. Open the full output data in a text editor
(array([[[2.10084091e-03, 4.23669480e-02, 2.66806763e-02],
         [5.94570711e-02, 1.06186241e-01, 9.00943354e-02],
         [4.49042171e-02, 1.10247076e-01, 8.91686454e-02],
         ...,
         [5.04519939e-02, 1.61853194e-01, 8.97769034e-02],
         [6.84009939e-02, 1.62445128e-01, 9.33482274e-02],
         [5.25210227e-04, 5.44117726e-02, 3.67647153e-03]],
 
        [[3.61694768e-02, 9.11064595e-02, 6.91176653e-02],
         [1.15537956e-01, 1.79561064e-01, 1.54543564e-01],
         [1.28245682e-01, 2.06615016e-01, 1.76815435e-01],
         ...,
         [1.35866135e-01, 2.56589293e-01, 1.83155373e-01],
         [1.30541787e-01, 2.37789705e-01, 1.65214270e-01],
         [1.17997229e-02, 1.12990208e-01, 4.16316614e-02]],
 
        [[8.24229717e-02, 1.59593850e-01, 1.25595242e-01],
         [1.67226911e-01, 2.57540226e-01, 2.18484759e-01],
         [1.84100837e-01, 2.83959389e-01, 2.40979701e-01],
         ...,
         [1.46407828e-01, 2.85238564e-01, 2.07157284e-01],
         [1.41622454e-01, 2.62870669e-01, 1.90549254e-01],
         [2.35294141e-02, 1.34663880e-01, 6.53711557e-02]],
 
        ...,
...
         ...,
         [3.34491432e-02, 1.32832348e-01, 8.18519518e-02],
         [2.41179056e-02, 1.17739268e-01, 6.67588711e-02],
         [7.35198334e-03, 9.57270637e-02, 4.47466671e-02]]], dtype=float32),
 'australian_terrier')
```

Now we've got ways to get:
* Prediction labels
* Validation labels
* Validation images

Lets make some function to make these all a bit more visualise.

We'll create a function which:

* Takes an array of prediction probabilities, and array of truth labels and an array of images and integers ✔
* Convert the prediction probabilitites to a predicted label ✔
* Plot the predicted labels, its predicted probability, the truth label and the target image on a single plot ✔

```
def plot_pred(prediction_probabilities, labels , images, n=1):
    '''
    View the prediction, ground truth and image for sample n
    '''
    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

#     pred_prob, true_label, image = prediction_probabilities[n], unique_breeds[labels[n].argmax()], images[n]

    pred_label = get_pred_label(pred_prob)

    # PLot image and remove ticks
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    
    # Change the colour of the title depedning on whether the prediction is right or wrong
    if(pred_label == true_label):
        color = "green"
    else:
        color = "red"
    
    
    # Change plot title to be predicted, probability of prediction and truth label
    plt.title("{} {:2.0f}% {}".format(pred_label.title(),
                                             np.max(pred_prob)*100, true_label.capitalize())
                                      , color=color)
```

```
plot_pred(prediction_probabilities=predictions, labels=val_labels, images=val_images, n=17)
```

![image](https://github.com/allansanyaz/dog-vision/assets/87567534/d7a9200b-cddc-45b1-ab06-d6019d122040)

Now we've got one function to visualise our models top predictions, lets make another to view our moedls top 10 predictions

This function will:
* Take an input of a prediction probabilities array and a ground truth array integer ✔
* Find a prediction using `get_pred_label()` ✔
* Find top 10:
    * Prediction probabilities indexes ✔
    * Prediction probabilities values ✔
    * Predicition labels ✔
    
* Plot the top 10 predictition probability values and labels, coloring the true label green.
    
```
def plot_pred_conf(prediction_probabilities, images, labels, n=1):
    '''
    Plot the top 10 highest prediction confidences along with the truth label for sample n.
    '''
    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]
    
    # Get the proedicted label
    pred_label = get_pred_label(pred_prob)
    
    # Find the top 10 prediction confidence indices
    top_10_pred_indexes = np.argsort(pred_prob)[-10:][::-1]
    # Find the top 10 predictions confidence values
#     top_10_pred_values = [pred_prob[index] for index in top_10_pred_indexes] # alternate below
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    # Find the top 10 prediction labels
#     top_10_pred_labels = [unique_breeds[index] for index in top_10_pred_indexes]
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]
    
    # Setup the plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,8))
    top_plot = ax[0].bar(top_10_pred_labels,
                  top_10_pred_values, color="grey")
    ax[0].set_xticklabels(top_10_pred_labels, rotation=90, fontsize=14)
    y_ticks = np.round(np.arange(0, 1.1, 0.2), decimals=2)
    ax[0].set_yticks(y_ticks)
    ax[0].set_yticklabels(y_ticks, fontsize=14)
    
    # Also be done using **************************************************
#     top_plot = plt.bar(np.arange(len(top_10_pred_labels)), 
#                        top_10_pred_values, color="grey")
#     plt.xticks(np.arange(len(top_10_pred_labels)), 
#                labels=top_10_pred_values, rotation="vertical")
    
    # Change the color of the true label
    if np.isin(true_label, top_10_pred_labels):
        top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
    else:
        pass

    ax[1].imshow(image)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
```

```
plot_pred_conf(predictions, val_images, val_labels, 200)
```

![image](https://github.com/allansanyaz/dog-vision/assets/87567534/2b700bb9-cf8d-4f20-ac2f-2bc1fa4e963a)

Now we've got some function to help us to visualise our predictions and evaluate our model, lets check out a few

```
def plot_pred_other(prediction_probabilitites, labels, n=1):
    '''
    Plot the top 10 highest prediction confidences along with the truth label for sample n.
    '''
    pred_prob, true_label = prediction_probabilitites[n], labels[n]
    
    # Get the proedicted label
    pred_label = get_pred_label(pred_prob)
    
    # Find the top 10 prediction confidence indices
    top_10_pred_indexes = np.argsort(pred_prob)[-10:][::-1]
    # Find the top 10 predictions confidence values
#     top_10_pred_values = [pred_prob[index] for index in top_10_pred_indexes] # alternate below
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    # Find the top 10 prediction labels
#     top_10_pred_labels = [unique_breeds[index] for index in top_10_pred_indexes]
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]
    
    # Setup the plot
    top_plot = plt.bar(np.arange(len(top_10_pred_labels)), 
                       top_10_pred_values, color="grey")
    plt.xticks(np.arange(len(top_10_pred_labels)), 
               labels=top_10_pred_labels, rotation="vertical")
    
    # Change the color of the true label
    if np.isin(true_label, top_10_pred_labels):
        top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
    else:
        pass
```

```
plot_pred_other(predictions, val_labels, 0)
```

![image](https://github.com/allansanyaz/dog-vision/assets/87567534/9b403164-1a96-4c9b-9cc2-b7679f735d72)

```
# Let's check out a few predictions and their different values
i_multiplier = 0
num_rows = 3
num_cols = 2
num_images = num_rows * num_cols

plt.figure(figsize=(13 * num_cols, 8 * num_rows))
for i in range(num_images):    
    plt.subplot(num_rows, 2 * num_cols, 2 * i+1)
    plot_pred(prediction_probabilities=predictions,
             labels=val_labels,
             images=val_images,
             n=i+i_multiplier)
    
    plt.subplot(num_rows, 2 * num_cols, 2 * i+2)
    plot_pred_other(prediction_probabilitites=predictions, 
                    labels=val_labels, 
                    n=i+i_multiplier)

plt.tight_layout(h_pad=1.0)
plt.show()
```

![image](https://github.com/allansanyaz/dog-vision/assets/87567534/69f950c2-0c09-4d7e-b8c5-6f6ba564aeb7)

```
i_multiplier=0
for i in range(num_images):
    plt.figure(figsize=(8 * num_cols, 13 * num_rows))
    
    plt.subplot(2 * num_rows, num_cols, i_multiplier+1)
    plot_pred(prediction_probabilities=predictions,
             labels=val_labels,
             images=val_images,
             n=i+i_multiplier)
    
    plt.subplot(2* num_rows, num_cols, i_multiplier+2)
    plot_pred_other(prediction_probabilitites=predictions, 
                    labels=val_labels, 
                    n=i+i_multiplier)
    
    i_multiplier+=2

plt.tight_layout(h_pad=1.0)
plt.show()
```

This shows the images and probability for the breeds

![image](https://github.com/allansanyaz/dog-vision/assets/87567534/2e661cf3-e830-470b-8bb8-6ae9f53ab083)
![image](https://github.com/allansanyaz/dog-vision/assets/87567534/2a04beb6-3716-477b-a0fb-1e5291aa8a7c)
![image](https://github.com/allansanyaz/dog-vision/assets/87567534/c8b8bdaf-f5cc-4f54-b8e0-2cf855cbfa4b)
![image](https://github.com/allansanyaz/dog-vision/assets/87567534/205a2d4d-2e2f-4652-aad0-f710cbda8f26)
![image](https://github.com/allansanyaz/dog-vision/assets/87567534/beccefb9-29b8-4365-856e-58f701060201)
![image](https://github.com/allansanyaz/dog-vision/assets/87567534/d8826a19-e5f4-4ae7-b768-a82cfa2c38a7)

**Challenge:** how would you create a confusion matrix of our models predictions and true labels?
```
tf.math.confusion_matrix(y_val[99], predictions[99])
```
** Output:
```
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[119,   0],
       [  1,   0]])>
```

### Saving and reloadind a trained model

```
# Create a function to save a model
def save_model(model, suffix=None):
    '''
    Saves a given model in a models directory and appends suffix (string)
    '''
    # Create a model directory pathname with current time
    modeldir = os.path.join(f"{dogvision_path}models",
                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_path = f"{modeldir}-{suffix}" # save format of our model
    print(f"saving model to: {model_path}...")
    model.save(model_path)
    
    return model_path
```

```
# Create a function to load a trained model
def load_model(model_path):
    ''''
    Loads a saved model from a specified path
    '''
    print(f"Loading a saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path,
                                      custom_objects={"KerasLayer": hub.KerasLayer})
    
    return model
```

Now we've got functions to save and load a trained model
* Save our model trained on 1000 images
```
save_model(model, suffix="5000-images-mobilenetv2-Adam")
```
** Output:
```
saving model to: models\20210223-021912-5000-images-mobilenetv2-Adam...
INFO:tensorflow:Assets written to: models\20210223-021912-5000-images-mobilenetv2-Adam\assets
INFO:tensorflow:Assets written to: models\20210223-021912-5000-images-mobilenetv2-Adam\assets
'models\\20210223-021912-5000-images-mobilenetv2-Adam'
```

Load a trained model
```
loaded_5000_image_model = load_model("models{0}20210223-021912-5000-images-mobilenetv2-Adam".format(os.sep))
```
** Output:
```
Loading a saved model from: models\20210223-021912-5000-images-mobilenetv2-Adam
```

Evaluate the pre-saved model
```
model.evaluate(val_data)
```
** Output:
```
32/32 [==============================] - 4s 116ms/step - loss: 0.7179 - accuracy: 0.7740
[0.7178705334663391, 0.7739999890327454]
```

Evaluate the loaded model
```
loaded_5000_image_model.evaluate(val_data)
```
** Output:
```
32/32 [==============================] - 5s 114ms/step - loss: 0.8441 - accuracy: 0.7740
[0.8441308736801147, 0.7739999890327454]
```

## Training a big dog model🐶 (on the full data)

```
len(X), len(y)
```
** Output:
```
(10222, 10222)
```

Create a data batch with the full data set
```
full_data = create_data_batches(X,y)
```

** Output:
```
Creating training data bacthes
```

```
full_data
```
** Output:
```
<BatchDataset shapes: ((None, 224, 224, 3), (None, 120)), types: (tf.float32, tf.bool)>
```

Create a model for full model
```
full_model = create_model()
```
** Output:
```
Building model with: https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
```

Create full model call backs
```
full_model_tensorboard = create_tensorboard_callback()
```
No validation set training on all the data, so we can't monitor validation accuracy
```
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=3)
```

**Note:** Running the cell below will take a full while (maybe up to 30 minutes for the epoch) because the GPU we're using in the runtime has to load all the images into the memory

Fit the full model to the data
```
full_model.fit(x=full_data, epochs=NUM_EPOCHS, 
               callbacks=[full_model_tensorboard, full_model_early_stopping])
```

** Output:
```
Output exceeds the size limit. Open the full output data in a text editor
Epoch 1/100
320/320 [==============================] - 130s 385ms/step - loss: 2.2966 - accuracy: 0.4927
Epoch 2/100
320/320 [==============================] - 37s 116ms/step - loss: 0.3763 - accuracy: 0.8942
Epoch 3/100
320/320 [==============================] - 37s 116ms/step - loss: 0.2248 - accuracy: 0.9456
Epoch 4/100
320/320 [==============================] - 37s 115ms/step - loss: 0.1466 - accuracy: 0.9651
Epoch 5/100
320/320 [==============================] - 40s 123ms/step - loss: 0.0905 - accuracy: 0.9836
Epoch 6/100
320/320 [==============================] - 38s 117ms/step - loss: 0.0688 - accuracy: 0.9903
Epoch 7/100
320/320 [==============================] - 37s 116ms/step - loss: 0.0522 - accuracy: 0.9946
Epoch 8/100
320/320 [==============================] - 38s 117ms/step - loss: 0.0437 - accuracy: 0.9941
Epoch 9/100
320/320 [==============================] - 38s 119ms/step - loss: 0.0326 - accuracy: 0.9971
Epoch 10/100
320/320 [==============================] - 37s 116ms/step - loss: 0.0266 - accuracy: 0.9984
Epoch 11/100
320/320 [==============================] - 37s 116ms/step - loss: 0.0251 - accuracy: 0.9981
Epoch 12/100
320/320 [==============================] - 37s 115ms/step - loss: 0.0206 - accuracy: 0.9986
Epoch 13/100
...
Epoch 23/100
320/320 [==============================] - 37s 116ms/step - loss: 0.0114 - accuracy: 0.9980
Epoch 24/100
320/320 [==============================] - 38s 117ms/step - loss: 0.0218 - accuracy: 0.9952
<tensorflow.python.keras.callbacks.History at 0x1cb9dc7c6d0>
```

```
save_model(full_model, suffix="full-image-set-mobilenetv2-Adam")
```

** Output:
```
saving model to: models\20210223-023929-full-image-set-mobilenetv2-Adam...
INFO:tensorflow:Assets written to: models\20210223-023929-full-image-set-mobilenetv2-Adam\assets
INFO:tensorflow:Assets written to: models\20210223-023929-full-image-set-mobilenetv2-Adam\assets
'models\\20210223-023929-full-image-set-mobilenetv2-Adam'
```

Loading the full model
```
loaded_full_model = load_model("models{}20210223-023929-full-image-set-mobilenetv2-Adam".format(os.sep))
```

** Output:
```
Loading a saved model from: models\20210223-023929-full-image-set-mobilenetv2-Adam
```

## Making predictions on the test data set

Since our model has been trained on images in the form of tensor batches. To make predictions on the test data we'll have to get it into the same format.

Luckily we created `create_data_batches()` earlier which can take a list of filenames as input and convert them into TensorBatches.

To make predictions on the test data we'll:
* Get the test image filenames ✔
* Convert the filenames into test data batches using `create_data_batches` and setting the `test_data` parameter to `True` (Since the test data doesn't have any labels).
* Make predictions array by passing the test batches to the `predict()` method called on our model

Load image filenames
```
test_path = "dog-breed-identification{0}test".format(os.sep)
test_filenames = [f"{test_path}{os.sep}{test_image}" for test_image in os.listdir(test_path)]
len(test_filenames)
```

** Output:
```
10354
```

```
test_data = create_data_batches(test_filenames, test_data=True)
test_data
```
** Output:
```
<BatchDataset shapes: (None, 224, 224, 3), types: tf.float32>
```

**Note:** Calling `predict()` onour full model and passing it the test data will take a long time to run (about an hour 1hr)

```
%time
# Make predictions on test data batch using the loaded full model
test_predictions = full_model.predict(test_data, verbose=1)
```

** Output:
```
Wall time: 0 ns
324/324 [==============================] - 37s 114ms/step
```

```
test_predictions[22]
```

** Output:
```
Output exceeds the size limit. Open the full output data in a text editor
array([2.2304509e-11, 5.6929047e-16, 6.1680266e-12, 4.1122251e-16,
       1.1935333e-10, 1.8258675e-13, 4.1818587e-10, 1.2358131e-14,
       2.4194370e-15, 2.0907960e-18, 9.6411547e-14, 1.8152284e-16,
       1.7183775e-13, 2.7900518e-14, 5.1132092e-20, 2.9713546e-14,
       2.9107599e-15, 5.3396949e-15, 1.2233680e-16, 1.2853490e-06,
       1.1215576e-12, 5.2015330e-16, 3.8040266e-13, 2.8147582e-13,
       7.4490824e-18, 4.8489379e-15, 3.4930383e-15, 4.7657066e-13,
       4.2986569e-12, 4.9950131e-11, 2.4519625e-14, 3.1117280e-13,
       6.4920124e-15, 5.5109323e-10, 2.2823368e-09, 9.4096470e-14,
       1.6748626e-19, 1.2164773e-14, 7.1652107e-12, 4.8516692e-14,
       2.0641099e-14, 3.6600566e-16, 3.2582650e-13, 4.9434405e-16,
       7.4355667e-11, 4.0226780e-12, 1.6658326e-13, 1.7388944e-14,
       6.6288283e-14, 7.3395888e-17, 3.7633794e-18, 8.8976379e-13,
       2.0916208e-15, 1.9367034e-19, 1.6597418e-11, 2.8702318e-12,
       1.9503861e-15, 1.2379255e-16, 1.2389242e-13, 9.1014449e-14,
       7.7410863e-07, 7.1784308e-14, 5.4153844e-13, 1.4819345e-11,
       7.5063084e-13, 1.6776580e-14, 4.2831063e-16, 1.9202080e-10,
       6.5132704e-15, 1.5086652e-16, 8.7475335e-14, 3.1124771e-16,
       2.2280942e-12, 4.4707674e-15, 9.9999762e-01, 8.7860847e-12,
       2.0259100e-13, 1.5477474e-13, 2.6237671e-14, 5.4185768e-13,
       2.7043388e-11, 6.3761576e-18, 4.5320493e-16, 1.4548547e-15,
       3.5008192e-13, 4.2332391e-16, 7.2179780e-16, 1.0491947e-11,
       9.1953476e-12, 1.8193608e-08, 1.1224635e-11, 4.8158253e-17,
       9.7738286e-19, 3.9428335e-16, 3.2793394e-16, 1.7447434e-08,
       1.8353870e-15, 2.0954209e-09, 3.6931916e-12, 2.1170756e-15,
...
       2.0319531e-07, 2.6292325e-14, 9.2206158e-13, 4.0900843e-14,
       5.7520823e-13, 3.6848123e-15, 2.4740399e-12, 1.5042426e-13,
       6.6630477e-12, 2.6983154e-15, 4.1417242e-10, 1.0951197e-15,
       3.3392656e-15, 5.5383960e-09, 2.4543254e-14, 3.6807566e-13],
      dtype=float32)
```

Save predictions (NumPy array) to csv file (for access later)
```
np.savetxt("predictions_array.csv", test_predictions, delimiter=",")
```

Load predictions (NumPy array) from csv file
```
loaded_predictions = np.loadtxt("predictions_array.csv", delimiter=",")
```

```
test_predictions[22] == loaded_predictions[22]
```

** Output:
```
array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True])
```

## Setting up our data for kaggle submission

We find that Kaggle wants our models prediction probability in a DataFrame with an ID as a column for each different dog breed.
"https://www.kaggle.com/c/dog-breed-identification/overview/evaluation"

To get the data in this format, we'll:
* Create a pandas DataFrame with an ID column as well as a column for each dog breed
* Add data to the ID column by extracting the test image ID's from their filepaths
* Add data (the prediction probabilities) to each of the dog breeds
* Export the data frame as a CSV and submit to Kaggle

## Method 1

```
# Get list of indexes
index = [os.path.splitext(fname.split(os.sep)[-1])[0] for fname in test_filenames]
# Create DataFrame with ids as index column and the unique breeds as the rest of columns
test_frame = pd.DataFrame(test_predictions, columns=unique_breeds) # could also pass index parameter as index
# Rename the index column
test_frame.insert(0, column="id", value=index)
```

```
def test_dog_predictor(predictions, index=0):
    print(predictions[index])
    print(f"Max value (probability of prediction): {np.max(predictions[index])}")
    print(f"Sum: {np.sum(predictions[index])}")
    print(f"Max index: {np.argmax(predictions[index])}")
    print(f"Predicted label: {unique_breeds[np.argmax(predictions[index])]}")
```

** Output:
```
Output exceeds the size limit. Open the full output data in a text editor
[3.70405475e-13 1.82821342e-13 1.79593947e-17 5.51879605e-16
 1.58403987e-14 2.05079267e-13 2.12096437e-19 1.50949837e-15
 1.10101599e-15 1.52785055e-12 1.71463318e-16 2.36957345e-13
 3.41687297e-14 1.26462009e-11 1.00171744e-16 2.79099474e-14
 5.48067883e-15 2.15123075e-16 1.21836596e-13 8.18611021e-13
 8.73996851e-17 5.07928321e-14 2.02888235e-15 8.56811519e-16
 6.62471615e-16 4.37054960e-13 1.31725090e-15 1.25862623e-18
 5.91039490e-18 1.88434107e-13 1.20460748e-14 6.81499263e-14
 1.22937413e-14 2.73303602e-11 2.74815009e-17 7.45552052e-19
 1.14935297e-13 5.08659498e-17 4.93157185e-14 2.00698735e-10
 1.53089835e-13 8.87060019e-16 1.22188416e-14 1.29295556e-16
 9.09303286e-14 4.65885941e-11 1.58097022e-14 1.66825931e-13
 3.10646149e-14 1.33512803e-13 5.00556280e-15 7.74884226e-13
 9.88752700e-13 5.75200963e-16 2.03526313e-14 1.35557366e-17
 5.00362909e-17 7.01411731e-16 1.13995142e-14 3.20198670e-17
 1.19309620e-15 9.99999881e-01 5.57769968e-15 1.18864358e-16
 1.70953992e-15 1.03118009e-16 4.97043486e-16 3.97851240e-13
 1.34267659e-17 3.26155817e-15 1.29743384e-13 4.17742907e-13
 1.70241936e-16 3.85527131e-15 1.52122723e-15 1.97068618e-17
 2.27901044e-16 3.79344552e-15 1.25265521e-15 5.32936132e-18
 1.55425583e-17 1.92873890e-18 6.80898340e-16 5.44809813e-15
 2.34670009e-11 1.13929687e-07 6.69871242e-17 1.43483194e-11
 1.54143666e-14 1.59395107e-17 5.81300289e-15 5.14082719e-16
 4.03496464e-13 2.24141831e-13 5.73432428e-14 1.00028702e-13
 1.37257414e-16 5.99132820e-16 2.82599855e-12 6.90194152e-11
...
Max value (probability of prediction): 0.9999998807907104
Sum: 1.0
Max index: 61
Predicted label: japanese_spaniel
```

## Method 2

Create a pandas dataframe
```
preds_df = pd.DataFrame(columns=["id"] + list(unique_breeds))
preds_df.head()
```

Append test image IDs to predictions DataFrame
```
test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df["id"] = test_ids
```

Add the prediction probabilities to each dog breed column
```
preds_df[list(unique_breeds)] = test_predictions
preds_df.head()
```

```
preds_df.to_csv("full_model_prediction_submission_1_mobilenetV2.csv", index=False)
```

### Making predictions on custom images, we'll:
* Get the file paths
* Turn the filepaths into data batches using `create_data_batches`. And since our custom images won't have labels, we set the `test_data` parameter to `True`
* Pass the custom image data batch to our moedls predict() method
* Convert the prediction probabilities to prediction labels
* Compre the predicted labels to the custom images

```
# The the custom image filepaths
custom_path="custom"
custom_image_paths = [f"{custom_path}{os.sep}{fname}" for fname in os.listdir(custom_path)]

# Obtain the custom data batches for predictions
custom_data = create_data_batches(custom_image_paths, test_data=True)
```

Predict the outcomes
```
custom_predictions = full_model.predict(custom_data)
test_dog_predictor(custom_predictions,1)
```

** Output:
```
Output exceeds the size limit. Open the full output data in a text editor
[2.61113103e-10 4.46699708e-12 2.09415649e-14 1.18806050e-12
 2.94328906e-12 2.04634183e-12 9.18295311e-16 1.20688656e-13
 1.20810886e-14 8.81618488e-15 4.07395309e-13 3.84408144e-10
 3.71379902e-14 1.23522321e-11 4.08678300e-11 5.32430437e-12
 6.84644386e-09 1.58391895e-13 8.03790715e-11 3.40475288e-13
 1.00473996e-09 9.46843190e-13 3.37036476e-14 2.17444465e-10
 2.25071125e-11 9.64901830e-08 1.39678062e-16 2.38640585e-15
 7.92375748e-11 8.89893323e-15 2.58690296e-08 2.15272904e-11
 1.21954953e-08 1.23798936e-08 2.33408309e-06 4.88666703e-12
 2.19114824e-10 2.15896067e-13 1.66320015e-13 5.91927386e-13
 2.68606821e-11 9.47542400e-09 2.36930202e-12 2.03298953e-13
 7.17631610e-06 7.70324397e-13 2.49525140e-12 1.96898886e-11
 1.91409200e-08 1.71625072e-06 3.46353417e-08 2.66109509e-12
 4.43440618e-09 1.66225679e-11 2.36526421e-09 3.56284419e-14
 1.25162519e-11 6.39630425e-14 9.81232109e-12 1.06620420e-12
 7.86487495e-17 5.06489945e-12 7.42130424e-13 1.77021505e-12
 2.36692380e-14 6.75318690e-11 2.50384624e-09 2.93140928e-10
 5.53604607e-14 5.06278539e-08 9.63056232e-11 1.13178973e-10
 1.74484830e-13 9.99568843e-15 8.58421895e-13 3.01650713e-13
 7.99090611e-14 9.37729529e-16 9.99987841e-01 2.03892689e-13
 1.29269848e-12 1.17847164e-11 9.26263623e-13 1.88969596e-09
 2.00611554e-12 5.48742336e-11 3.65312191e-13 1.12087726e-12
 1.02718346e-11 1.28771028e-12 2.37472902e-10 8.81317727e-11
 2.42304274e-08 7.26359424e-12 3.93209380e-12 1.00671771e-09
 1.77755963e-13 1.22191396e-10 2.98532412e-13 3.16787720e-12
...
Max value (probability of prediction): 0.9999878406524658
Sum: 1.0
Max index: 78
Predicted label: newfoundland
```

```
custom_image_paths[1]
```
** Output:
```
'custom\\newfoundland_dog_pictures.jpg'
```

```
custom_pred_label = [get_pred_label(custom_predictions[i]) for i in range(len(custom_predictions))]
custom_pred_label
```
** Output
```
['golden_retriever', 'newfoundland', 'border_collie']
```

Get custom images (our unbatchify()) function won't work since there arent labels... maybe we could fix this later)
```
custom_images = []
# loop through unbatched data
for image in custom_data.unbatch().as_numpy_iterator():
    custom_images.append(image)
```

Check custom image predictions
```
plt.figure(figsize=(10,10))
for i, image in enumerate(custom_images):
    plt.subplot(1, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title(custom_pred_label[i])
    plt.imshow(image)
```

![image](https://github.com/allansanyaz/dog-vision/assets/87567534/18d91466-4a83-41f0-803d-991157643b49)















