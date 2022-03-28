"""Train model"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, explained_variance_score
import pandas as pd
import numpy as np
from joblib import dump
import os

# Models types
_LDA = 'lda'
_RF = 'random_forest'
_SVR = 'svr'

# Folder to save models to
SAVE_DIR = 'models'
# Path to normalized data
DATA_PATH = 'processed.csv'


def evaluate(model, test_features, test_labels):
    """
    Calculate model metrics and print them

    :param model: Trained model to evaluate
    :param test_features: Test set of values
    :param test_labels: Test set of labels
    :return: Explained variance scoring of model
    """
    predictions = model.predict(test_features)
    mae = mean_absolute_error(test_labels, predictions)
    ev = explained_variance_score(test_labels, predictions) * 100
    print('Model Performance')
    print('MAE: {:0.4f}.'.format(mae))
    print('Explained variance = {:0.2f}%.'.format(ev))

    return ev


def get_grid(model_type=_RF):
    """
    Get grid of model parameters to test

    :param model_type: Parameters for this type of model will be returned
    :return: Grid of parameters
    """

    if model_type == _LDA:
        grid = {
            'shrinkage': np.arange(0, 1, 0.01),
            'solver': ['svd', 'lsqr', 'eigen']
        }
        return grid
    elif model_type == _RF:
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=50, stop=150, num=3)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt', 'log2']
        # Maximum number of levels in tree
        max_depth = [10, 30, None]
        # Minimum number of samples required to split a node
        min_samples_split = [2, 4]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 4, 8]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }
        return grid
    elif model_type == _SVR:
        grid = {
            'kernel': ('linear', 'rbf'),
            'C': [1, 10, 100]
        }
        return grid
    else:
        return {}


def model_type_to_model(model_type):
    """
    Return model from sklearn by its name

    :param model_type: Name of the model type
    :return: Model from sklearn
    """
    if model_type == _RF:
        return RandomForestRegressor()
    elif model_type == _SVR:
        return SVR()
    elif model_type == _LDA:
        return LinearDiscriminantAnalysis()
    else:
        raise ValueError("Model type is incorrect! Use models defined in module")


def model_searching(X, y, output, model_type=_RF):
    """
    Search the best parameters of this model, train it and save

    It is splitting values on train and test set, so no need to evaluate after

    :param X: Values
    :param y: Labels
    :param output: Path to save model to
    :param model_type: Name of model type
    """
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    base_model = model_type_to_model(model_type)
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)

    # Use the grid to search for best hyperparameters
    grid = get_grid(model_type)

    # First create the base model to tune
    model = model_type_to_model(model_type)

    # Random search of parameters, using 3-fold cross validation,
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=3, n_jobs=-1)

    # Fit the random search model
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, X_test, y_test)
    dump(best_grid, output)
    print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))


def get_data():
    """
    Read all data

    :return: X, y where X - values, y - labels
    """
    # Read all data
    data = pd.read_csv(DATA_PATH)
    y = data['diagnosis']
    X = data.copy()
    X.drop('diagnosis', axis=1, inplace=True)
    return X, y


X, y = get_data()
for model_type in [_RF, _SVR, _LDA]:
    print(f'Model {model_type}')

    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    output = os.path.join(SAVE_DIR, f'{model_type}.joblib')
    model_searching(X, y, output, model_type)
