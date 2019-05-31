'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import model_selection, metrics, neighbors, naive_bayes, svm, tree


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''

    return [(9956522, 'Nam', 'Nguyen'), (9809589, 'Dylan', 'Pryke-Watanabe'), (10008217, 'Texas', 'Barnes')]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    raw_data = np.genfromtxt(dataset_path, delimiter=',', dtype=None)
    X = []
    y = []
    for row in raw_data:
        X.extend([list(row)[2:]])
        if row[1] == b'M':
            y.extend([1])

        else:
            y.extend([0])
    X = np.array(X)
    y = np.array(y)
    return (X, y)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Creates Decision Tree classifier
    tree_clf = tree.DecisionTreeClassifier()

    # Set the parameters to be compared in the grid search
    params = [{'max_depth': np.linspace(1, 100, 100)}]

    # Finds the best parameter for the classifier
    clf = model_selection.GridSearchCV(tree_clf, params)

    # Train the Decision Tree Classifier
    clf.fit(X_training, y_training)

    # Output the trained Decision Tree
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    # TODO: dylan
    raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    # TODO: dylan
    raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def neural_model(neurons):
    '''
    Builds the model for the neural network of which the classifier will use 
    '''

    #Creates the model
    neural = Sequential()

    #Adds Input and hidden layer 1
    neural.add(Dense(neurons, input_dim=30, activation='relu'))

    #Hidden layer 2
    neural.add(Dense(neurons, activation='relu'))

    #Output layer
    neural.add(Dense(1, activation='sigmoid'))

    #Compiles the model
    neural.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return neural


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network with two dense hidden layers classifier 
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''

    # Creates Neural Network Classifier with neural_model as a basis
    neural_clf = KerasClassifier(build_fn=neural_model)

    # Set the parameters to be compared in grid search
    params = [{'neurons': [1, 5, 10, 20, 30]}]

    # Finds the best parameter for the classifier
    clf = model_selection.GridSearchCV(neural_clf, params)

    # Train the neural network
    clf = clf.fit(X_training, y_training)

    # Output the trained network
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    (X_training, y_training) = prepare_dataset('medical_records.data')

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_training, y_training, test_size=0.33)

    #clf_tree = build_DecisionTree_classifier(X_train, y_train)
    #y_pred_tree = clf_tree.predict(X_test)
    #print("Accuracy", metrics.accuracy_score(y_test, y_pred_tree))


    clf_neural = build_NeuralNetwork_classifier(X_train, y_train)
    y_pred_neural = clf_neural.predict(X_test)
    print("Accuracy", metrics.accuracy_score(y_test, y_pred_neural))
