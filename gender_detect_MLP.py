import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, Activation
from keras import applications
from keras.utils.np_utils import to_categorical
import os
import matplotlib.pyplot as plt
import argparse


class Image_MLP(object):

    '''
    Image_MLP

    Parameters
    ____________
    img_width : Int
        the width of the images you will be classifying
    img_height : Int
        the height of the images you will be classifying
    (optional) model_path : String
        the path to the saved model you'd like to use
    Attributes
    ____________
    model : keras model
        The model after fitting
    img_width : Int
        the width of the images the model classifies
    img_height : Int
        hte height of the images the model classifies
    '''
    def __init__(self,
                 img_width = 640,
                 img_height = 480,
                 model_path = None):
        self.img_width = img_width / 10
        self.img_height = img_height / 10
        if model_path != None:
            self.set_model(model_path)
        else:
            self.model = None

    '''
    Set_Model

    Sets the MLP model to the model at the model path.

    Parameters
    ____________
    model_path : String
        the path of the model which you are setting
    '''
    def set_model(self,
                  model_path):
        self.model = load_model(model_path)

    '''
    Save_Model

    Saves the current MLP model to save_path.

    Parameters
    ____________
    save_path : String
        the path to which you'd like to save the model (must end in '.h5')
    '''
    def save_model(self,
                   save_path = 'saved_models/MLP_model.h5'):
        self.model.save(save_path)


    """
    fit

    fits the MLP to the input data and labels. Optimizes using gradient descent,
    and then evaluates the fitted model on the data in data/validation.

    Parameters
    ---------
    data_dir : String
        the path to the data master directory
    (optional) batch_size : int
        the size of the batches used for gradient descent optimization.
    (optional) epochs : int
        the number of iterations in which the model evaluates a batch to
        optimize
    Return
    ------
    self : object
        self is fitted
    """
    def fit(self,
            data_dir = './data',
            batch_size = 1,
            epochs = 100):

        train_data_dir = data_dir + '/train/'
        validation_data_dir = data_dir + '/validation/'

        train_labels = []
        for filename in os.listdir(train_data_dir + "male/"):
            train_labels.append(1)
        for filename in os.listdir(train_data_dir + "female/"):
            train_labels.append(0)
        train_labels = to_categorical(train_labels[0: \
        batch_size * int(len(train_labels) / batch_size)])
        nb_train_samples = len(train_labels)

        datagen = ImageDataGenerator(rescale=1. / 255,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True)

        # build the VGG16 network
        model = applications.VGG16(include_top=False,
                                   weights='imagenet')

        generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
        train_data = model.predict_generator(
            generator,
            nb_train_samples / batch_size)

        validation_labels = []
        for filename in os.listdir(validation_data_dir + "male/"):
            validation_labels.append(1)
        for filename in os.listdir(validation_data_dir + "female/"):
            validation_labels.append(0)
        validation_labels = to_categorical(validation_labels[0: \
        batch_size * int(len(validation_labels) / batch_size)])
        nb_validation_samples = len(validation_labels)

        generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
        validation_data = model.predict_generator(
            generator,
            nb_validation_samples / batch_size)


        # Define the model
        model = Sequential()
        model.add(Flatten(input_shape=(train_data.shape[1:])))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='linear'))
        model.add(Dense(2, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        # The function to optimize is the cross entropy between the true label and the output (softmax) of the model
        # We will use adadelta to do the gradient descent see http://cs231n.github.io/neural-networks-3/#ada
        model.compile(loss='categorical_crossentropy',
                      optimizer='adamax',
                      metrics=["accuracy"])


        model_info = model.fit(train_data,
                               train_labels,
                               epochs=epochs,
                               batch_size=batch_size,
                               validation_data=(validation_data, validation_labels))

        '''models the history of a trianing session'''
        def plot_model_history(model_history):
            fig, axs = plt.subplots(1,
                                    2,
                                    figsize=(15,5))
            # summarize history for accuracy
            axs[0].plot(range(1,
                              len(model_history.history['acc'])+1),
                        model_history.history['acc'])
            axs[0].plot(range(1,
                              len(model_history.history['val_acc'])+1),
                        model_history.history['val_acc'])
            axs[0].set_title('Model Accuracy')
            axs[0].set_ylabel('Accuracy')
            axs[0].set_xlabel('Epoch')
            axs[0].set_xticks(np.arange(1,
                                        len(model_history.history['acc'])+1),
                              len(model_history.history['acc'])/10)
            axs[0].legend(['train', 'val'],
                          loc='best')
            # summarize history for loss
            axs[1].plot(range(1,
                              len(model_history.history['loss'])+1),
                        model_history.history['loss'])
            axs[1].plot(range(1,
                              len(model_history.history['val_loss'])+1),
                        model_history.history['val_loss'])
            axs[1].set_title('Model Loss')
            axs[1].set_ylabel('Loss')
            axs[1].set_xlabel('Epoch')
            axs[1].set_xticks(np.arange(1,
                                        len(model_history.history['loss'])+1),
                              len(model_history.history['loss'])/10)
            axs[1].legend(['train', 'val'],
                          loc='best')
            plt.show()

        # plot model history
        plot_model_history(model_info)
        # compute test accuracy
        self.model = model


    '''
    evaluate

    evaluates a model's prediction performance on a validation data set against
    its set of labels

    Parameters
    ____________
    validation_data_dir : String
        the path to the validation data directory
    (optional) batch_size : Int
        the larger the batch_size, the faster. So, if you want it really fast,
        set batch_size = the number of testing files. Just make sure that
        batch_size is a divisor of the number of testing files. Otherwise, some
        files won't get evaluated.
    Returns
    ____________
    void
    '''
    def evaluate(self,
                 validation_data_dir = "./data/validation",
                 batch_size = 1):
        if self.model == None:
            raise ValueError("Please fit model or load model before evaluation")
        else:
            validation_labels = []
            for filename in os.listdir(validation_data_dir + "/male/"):
                validation_labels.append(1)
            for filename in os.listdir(validation_data_dir + "/female/"):
                validation_labels.append(0)
            validation_labels = to_categorical(validation_labels[0: \
            batch_size * int(len(validation_labels) / batch_size)])
            nb_validation_samples = len(validation_labels)

            predicted_class = self.predict(validation_data_dir)
            true_class = np.argmax(validation_labels,
                                   axis=1)
            num_correct = np.sum(predicted_class == true_class)
            accuracy = float(num_correct)/predicted_class.shape[0]
            print "Accuracy on validation data: %0.2f"%(accuracy * 100)

    '''
    predict

    makes predictions based off of a trained model (will produce 0 or 1 for
    each time point)

    Parameters
    ____________
    test_data_dir : String
        the directory of the data you'd like to predict
    (optional) batch_size : Int
        the larger the batch_size, the faster. So, if you want it really fast,
        set batch_size = the number of testing files. Just make sure that
        batch_size is a divisor of the number of testing files. Otherwise, some
        files won't get evaluated.
    Returns
    ____________
    predictions : 1d array
        discrete values (either a 0 or a 1 at each time point)
    '''
    def predict(self,
                test_data_dir="./data/test"):
        if self.model == None:
            raise ValueError("Please fit model or load model before prediction")
        else:
            nb_test_samples = 1

            m = applications.VGG16(include_top=False,
                                   weights='imagenet')
            datagen = ImageDataGenerator(rescale=1. / 255,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True)
            generator = datagen.flow_from_directory(
                test_data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=1,
                class_mode=None,
                shuffle=False)
            filenames = generator.filenames
            test_data = m.predict_generator(
                generator,
                nb_test_samples)
            result = self.model.predict(test_data)
            predicted_class = np.argmax(result,
                                        axis=1)
            return predicted_class

#a small unit test
if __name__ == "__main__":
    #argument parsing
    parser = argparse.ArgumentParser(description='A general binary image \
    classifier.')
    parser.add_argument("model_name", help="The name of the model you are \
    either evaluating, using to predict, or the name under which you are \
    saving a fitted model. Must end in '.h5'",
                    type=str)
    parser.add_argument("--fit", help="Fit a model to the data in \
    data/train, validate on the data in data/validation, and save the model \
    under the model_name argument. Type true after if you'd like to fit.")
    parser.add_argument("--evaluate", help="Evaluate the model under the \
    model_name argument using the data in data/validation. Type true after if \
    you'd like to evalutae.")
    parser.add_argument("--predict", help="Predict the data in test/test using \
    the model under the model_name argument. Type true after if you'd like to \
    predict.")
    args = parser.parse_args()

    if args.model_name == "" or args.model_name == None or (not \
    args.model_name[len(args.model_name) - 3:] == ".h5") :
        raise ValueError("You must specify a model_name ending in '.h5.'")

    #create the model
    mlp = Image_MLP(img_width = 640, img_height = 480)
    if args.fit:
        #fit it to the data
        mlp.fit(batch_size = 19, epochs = 25)
        #save the fitted model
        mlp.save_model(save_path = 'saved_models/' + args.model_name)
    if args.evaluate:
        #load and set the saved model
        mlp.set_model(model_path = "saved_models/" + args.model_name)
        #evaluate the model
        mlp.evaluate()
    if args.predict:
        #load and set the saved model
        mlp.set_model(model_path = "saved_models/" + args.model_name)
        #evaluate the model
        mlp.predict()
