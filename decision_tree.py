import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():

    print("Wine Quality Classification using Decision Tree")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")
    print("Setting seed to allow repeatability...")
    seed = set_random_seed(0)
    
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


    # Train Decision Tree
    print("\n=======================")
    print("TRAINING")
    print("=======================")
    print("Training a model for White Wines...")
    white_clf = DecisionTreeClassifier(criterion="entropy", splitter='best', random_state=seed,
        max_depth=10,
        min_samples_split=2)
    white_clf.fit(white_train_data, white_train_labels)

    print("Training a model for Red Wines...")
    red_clf = DecisionTreeClassifier(criterion="entropy", splitter='best', random_state=seed,
        max_depth=7,
        min_samples_split=9)
    red_clf.fit(red_train_data, red_train_labels)

    
    print("\n=======================")
    print("TESTING")
    print("=======================")
    print("Running training samples through White Model to test accuracy...")
    white_train_predictions = white_clf.predict(white_train_data)
    print("Running training samples through Red Model to test accuracy...")
    red_train_predictions = red_clf.predict(red_train_data)

    print("Running testing samples through White Model to test accuracy...")
    white_test_predictions = white_clf.predict(white_test_data)
    print("Running testing samples through Red Model to test accuracy...")
    red_test_predictions = red_clf.predict(red_test_data)

    
    print("\n=======================")
    print("RESULTS")
    print("=======================")

    # Compute the training accuracies of the models.
    print(f"White Training Accuracy: {accuracy(white_train_predictions, white_train_labels) * 100:.2f}%")
    print(f"Red Training Accuracy: {accuracy(red_train_predictions, red_train_labels) * 100:.2f}%")

    # Compute the testing accuracies of the models.
    print(f"\nWhite Testing Accuracy: {accuracy(white_test_predictions, white_test_labels) * 100:.2f}%")
    print(f"Red Testing Accuracy: {accuracy(red_test_predictions, red_test_labels) * 100:.2f}%")
       

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


def accuracy(prediction, label):
    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == label[i]: correct += 1
    return correct/len(prediction)


def set_random_seed(seed):
    '''Set random seed for repeatability.'''
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    return seed


if __name__ == '__main__':
    main()