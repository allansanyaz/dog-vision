#!/usr/bin/env python

##### By ######
### AllanSanyaz ###
#################

# Import neccessary tools
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# import logging
# tf.get_logger().setLevel(logging.ERROR)
# print("TF version:", tf.__version__)

# %%

# Import dog breeds
unique_breeds = np.array([i.strip("\n") for i in open("dog_breeds.txt").readlines()])
# Define image size,
IMG_SIZE = 224

# %%

def process_options():
    parser = argparse.ArgumentParser(description='Process an image and identify the breed of dog', epilog="Dog breed identifier")
    parser.add_argument('-i', '--input', dest='image', help='name of the dog breed image file')
    parser.add_argument('-v','--version', action='version', version='1.0')
    
    args = parser.parse_args()

    if(len(sys.argv) < 1):
        parser.print_usage()
        sys.exit(1)
    
    return args

# %%

def validate_arguments(args):
    """
    Validate the arguments the user has parsed into the script
    """

    image = args.image

    if(not os.path.exists(image)):
        raise ValueError("Image file {0} not found".format(image))
    
    return image

# %%

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


# %%

def get_image_label(image_path, label):
    '''
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image,label)
    '''

    image = process_image(image_path)

    return image, label


# %%

# Turn probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
    '''
    Turns an array of predictiion probabilites into a label.
    '''

    return unique_breeds[np.argmax(prediction_probabilities)], np.max(prediction_probabilities)*100


# %%

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
        data_batch = data.map(process_image).batch(BATCH_SIZE) # the slice data set is what is going to the function with map so the data becomes the parameter

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


# %%

# Create a function to load a trained model
def load_model(model_path):
    ''''
    Loads a saved model from a specified path
    '''
    print(f"Loading a saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path,
                                      custom_objects={"KerasLayer": hub.KerasLayer})
    
    return model

# %%

def test_dog_predictor(predictions, index=0):
    '''
    Used to validate the result and certainty
    '''
    
    print(predictions[index])
    print(f"Max value (probability of prediction): {np.max(predictions[index])}")
    print(f"Sum: {np.sum(predictions[index])}")
    print(f"Max index: {np.argmax(predictions[index])}")
    print(f"Predicted label: {unique_breeds[np.argmax(predictions[index])]}")

# %%

# Check custom image predictions
def image_output(custom_images, custom_pred_label):
    '''
    Displays the breed and confidence of estimate
    '''
    
    fig, ax = plt.subplots(figsize=(6,6))
    for i, image in enumerate(custom_images):
        breed = custom_pred_label[i][0]
        confidence = custom_pred_label[i][1]

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("{1}: {0}%".format(int(confidence), breed.replace("_", " ").title()))
        ax.imshow(image)
        plt.show()

# %%
def main():
    
    args = process_options()
    dog_image = validate_arguments(args)

    # Load a trained model
    model = load_model("breed_model{0}".format(os.sep))

    # The the custom image filepaths
    custom_image_paths = [dog_image]
    # Obtain the custom data batches for predictions
    custom_data = create_data_batches(custom_image_paths, test_data=True)
    # Predict the outcomes
    custom_predictions = model.predict(custom_data)
    custom_pred_label = [get_pred_label(custom_predictions[i]) for i in range(len(custom_predictions))]

    # Get custom images
    custom_images = []
    # loop through unbatched data
    for image in custom_data.unbatch().as_numpy_iterator():
        custom_images.append(image)

    image_output(custom_images, custom_pred_label)


# %%
if __name__ == "__main__":
    main()
