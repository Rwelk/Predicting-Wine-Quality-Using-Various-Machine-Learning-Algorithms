##VERY VERY IMPORTANT NOTE: Since the quality ranges from 3 - 9 I.E Does not include outliers
##The ranges of qualities for this project will be from 0-6###



import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

ROOT = os.path.dirname(os.path.abspath(__file__))
display_warnings = Sequential()

def main():

    print("\n\nWine Quality Classification using a Neural Network")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")
    print("Setting seed to allow repeatability...")
    seed = set_random_seed(4578)
    
    # load data
    white_file = np.loadtxt(os.path.join(ROOT, 'winequality-white.csv'), delimiter = ';', dtype=str)
    red_file = np.loadtxt(os.path.join(ROOT, 'winequality-red.csv'), delimiter = ';', dtype=str)
    
    print("Loading attributes...")
    attributes = white_file[0]
    for i in range(len(attributes)):
        attributes[i] = attributes[i].replace('"', '')
    
    print("Loading training labels...")
    white_train_labels = white_file[1:, -1].astype('int32')
    red_train_labels = red_file[1:, -1].astype('int32')

    print("Loading training data...")
    white_train_data = white_file[1:,0:-1].astype('float64')
    red_train_data = red_file[1:,0:-1].astype('float64')

    print("Randomly selecting samples for testing...")
    white_idx = np.random.choice(white_train_data.shape[0], 100, replace=False)
    red_idx = np.random.choice(red_train_data.shape[0], 100, replace=False)

    print("Splitting off sample labels for testing...")
    white_train_labels, white_test_labels = separate(white_train_labels, white_idx)
    red_train_labels, red_test_labels = separate(red_train_labels, red_idx)

    print("Splitting off sample data for testing...")
    white_train_data, white_test_data = separate(white_train_data, white_idx)
    red_train_data, red_test_data = separate(red_train_data, red_idx)

    
    # Change class labels from point score of 3 - 9 to 0 - 2, where 0 is bad, 1 is average, and 2 is great
    white_train_labels = adjust_labels(white_train_labels)
    white_test_labels = adjust_labels(white_test_labels)
    red_train_labels = adjust_labels(red_train_labels)
    red_test_labels = adjust_labels(red_test_labels)


    print("\n=======================")
    print("TRAINING")
    print("=======================")
    print("Training a model for White Wines...")
    white_clf = Sequential()
    white_clf.add(Dense(units = 1000, activation='sigmoid', name = 'hidden1', input_shape=(11,))) ##First Hidden Layer##
    white_clf.add(Dense(units = 500, activation='sigmoid', name = 'hidden2')) ##Second Hidden Layer##
    white_clf.add(Dense(units = 7, activation='sigmoid', name = 'Output'))

    white_clf.compile(loss = 'mse', 
        optimizer=SGD(learning_rate=0.01), 
        metrics=['accuracy'])

    white_clf.fit(white_train_data, white_train_labels, epochs = 50, batch_size = 100, verbose=0)


    print("Training a model for Red Wines...")
    red_clf = Sequential()
    red_clf.add(Dense(units = 1000, activation='sigmoid', name = 'hidden1', input_shape=(11,))) ##First Hidden Layer##
    red_clf.add(Dense(units = 750, activation='sigmoid', name = 'hidden2')) ##Second Hidden Layer##
    red_clf.add(Dense(units = 7, activation='sigmoid', name = 'Output'))

    red_clf.compile(loss = 'mse', 
        optimizer=SGD(learning_rate=0.01), 
        metrics=['accuracy'])

    red_clf.fit(red_train_data, red_train_labels, epochs = 50, batch_size = 100, verbose=0)


    print("\n=======================")
    print("TESTING")
    print("=======================")
    print("Running training samples through White Model to test accuracy...")
    white_train_predictions = white_clf.predict(white_train_data, verbose=0)
    print("Running training samples through Red Model to test accuracy...")
    red_train_predictions = red_clf.predict(red_train_data, verbose=0)

    print("Running testing samples through White Model to test accuracy...")
    white_test_predictions = white_clf.predict(white_test_data, verbose=0)
    print("Running testing samples through Red Model to test accuracy...")
    red_test_predictions = red_clf.predict(red_test_data, verbose=0)


    print("\n=======================")
    print("RESULTS")
    print("=======================")
    print(f"White Model Metrics on Training Data:")
    showMetrics(white_train_predictions, white_train_labels)
    print(f"\nRed Model Metrics on Training Data:")
    showMetrics(red_train_predictions, red_train_labels)

    print(f"\nWhite Model Metrics on Testing Data:")
    showMetrics(white_test_predictions, white_test_labels)
    print(f"\nRed Model Metrics on Testing Data:")
    showMetrics(red_test_predictions, red_test_labels)


def set_random_seed(seed):
    '''Set random seed for repeatability.'''
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def separate(arr, idx):
    keep = []
    remove = []
    for i in range(len(arr)):
        if i in idx:
            remove.append(arr[i])
        else:
            keep.append(arr[i])

    keep = np.array(keep)
    remove = np.array(remove)

    return keep, remove


def adjust_labels(labels):
    for i in range(len(labels)):
        if labels[i] < 5:
            labels[i] = 0
        elif labels[i] < 8:
            labels[i] = 1
        else:
            labels[i] = 2

    return labels


def showMetrics(res, labels):

    #Computes average for top guess
    pred = np.argmax(res, axis=1)

    print(f"Accuracy on Top Guess: {np.mean(pred == labels) * 100:.2f}%")

    #Compute accuracy for top 3 guesses
    pred3 = np.argsort(res, axis=1)[:,4:]
    correct = 0 

    for i in range(pred3.shape[0]):
        if labels[i] in pred3[0]:
            correct += 1

    print(f"Accuracy on Top 3 Guesses: {correct / pred3.shape[0] * 100:.2f}%")


    #Computes average difference between the predicted label and the correct label
    print(f"Average Difference Between Predicted and Actual Label: {np.ceil(np.mean(abs(pred - labels)))}")

    #Shows count for each label

    counts = np.zeros(3)

    for i in range(len(counts)):
        results = np.array(np.where(labels == i))
        results = results.flatten()
        np.add.at(counts, i, len(results))
        line = 'Bad' if i == 0 else 'Average' if i == 1 else 'Good'
        print(f"# of Occurences for {line} Quality: {counts[i]}")


if __name__ == '__main__':
    main()

    

