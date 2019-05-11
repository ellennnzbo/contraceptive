import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

def main():
    print('---------------------- RANDOM FOREST CLASSIFIER ----------------------')
    df = pd.read_csv('cmc.data', header=None)
    column_names = ['wife_age', 'wife_edu', 'husband_edu', 'children', 'wife_religion', 'wife_work', 'husband_occ', 'sol_index', 'media', 'contraceptive']
    df.columns = column_names
    df = preprocess_data(df)

    # train test split
    train_inputs, train_targets, test_inputs, test_targets = train_test_split(df, train_percentage=0.75)

    # compare random search training w/ no search
    random_model, best_random = random_search_training(train_inputs, train_targets, n_iters=100, n_folds=3)
    random_accuracy = evaluate(random_model, test_inputs, test_targets)
    base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    base_model.fit(train_inputs, train_targets)
    base_accuracy = evaluate(base_model, test_inputs, test_targets)
    print('Comparing with and without random search training')
    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy-base_accuracy)/base_accuracy))

    # grid search with cross validation
    # create grid based on best parameters from random search
    param_grid = new_grid(best_random)
    print('Parameter grid for grid search')
    pprint(param_grid)
    grid_model = grid_search_with_cv(param_grid, train_inputs, train_targets, n_folds=3)
    best_grid = grid_model.best_estimator_
    grid_accuracy = evaluate(best_grid, test_inputs, test_targets)
    print('Comparing grid search to baseline model')
    print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy-base_accuracy)/base_accuracy))
    print('Feature Importance: ', best_grid.feature_importances_)

def grid_search_with_cv(param_grid, train_inputs, train_targets, n_folds):
    """grid search with cross-validation"""
    print('Conducting grid search...')
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=n_folds, n_jobs=-1)
    grid_search.fit(train_inputs, train_targets)
    print('Best parameters from grid search')
    pprint(grid_search.best_params_)
    return grid_search

def new_grid(old_params):
    """takes best parameters from random search and widens the range of values for each parameter for grid search"""
    n_estimators = old_params['n_estimators']
    old_params['n_estimators'] = np.arange(n_estimators-100, n_estimators+101, 100).tolist()
    min_samples_split = old_params['min_samples_split']
    old_params['min_samples_split'] = np.arange(min_samples_split-2, min_samples_split+3, 2).tolist()
    min_samples_leaf = old_params['min_samples_leaf']
    old_params['min_samples_leaf'] = list(filter(lambda a: a != 0, np.arange(min_samples_leaf-1, min_samples_leaf+2, 1).tolist()))
    old_params['max_features'] = [2, 3]
    max_depth = old_params['max_depth']
    old_params['max_depth'] = np.arange(max_depth, max_depth+31, 10).tolist()
    old_params['bootstrap'] = [True]
    new_params = old_params
    return new_params

def evaluate(model, test_features, test_labels):
    """evaluate accuracy of model"""
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors/test_labels)
    accuracy = 100 - mape
    return accuracy

def random_search_training(train_inputs, train_targets, n_iters, n_folds):
    """select a sample of grid combinations to fit model on"""
    print('Conducting random search training...')
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=n_iters, cv=n_folds,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(train_inputs, train_targets)
    best_random = rf_random.best_params_
    print('Best parameters from random search training')
    pprint(best_random)
    return rf_random, best_random


def train_test_split(dataframe, train_percentage):
    dataframe['is_train'] = np.random.uniform(0, 1, len(dataframe)) <= train_percentage
    train, test = dataframe[dataframe['is_train']==True], dataframe[dataframe['is_train']==False]
    features = dataframe.columns[:9]
    target = 'contraceptive'
    train_inputs = train[features]
    train_targets = train[target]
    test_inputs = test[features]
    test_targets = test[target]
    print('FEATURES: ', features)
    print('TARGET: ', target)
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:',len(test))
    return train_inputs, train_targets, test_inputs, test_targets

def preprocess_data(dataframe):
    """standardise age"""
    dataframe['wife_age'] -= dataframe['wife_age'].mean()
    dataframe['wife_age'] /= dataframe['wife_age'].std()
    return dataframe

if __name__ == '__main__':
    main()

