import matplotlib.pyplot as plt
import random
import datetime, os, time
import io
import seaborn as sns
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
import sklearn
from keras import backend as K
from keras.models import Model
from tensorflow.keras import layers
from keras.initializers import glorot_normal,  he_normal
from tensorflow import keras
from tensorflow.keras.layers import SimpleRNN, Flatten, Input, Dense, LSTM, RepeatVector, Bidirectional, Masking, Dropout, Layer, BatchNormalization, Conv1D
from google.colab import drive

from tensorflow.python.framework.ops import disable_eager_execution
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, GroupShuffleSplit, GridSearchCV, RandomizedSearchCV,RepeatedStratifiedKFold, train_test_split
from sklearn.svm import SVC

drive.mount('/content/drive')

from tensorflow.keras.layers import ConvLSTM1D, Flatten, Input, Dense, Lambda, LSTM, RepeatVector, Bidirectional, Masking, Dropout, Layer, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, ConvLSTM3D, Conv1D, ConvLSTM2D, MaxPool1D, AvgPool1D, Conv2D, AvgPool2D
from google.colab import drive
from tensorflow.python.framework.ops import disable_eager_execution
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras
from sklearn.decomposition import PCA



def plot_rul_one_engine(engine_number, functions_list):
    """Predicts and plots rul for each each timestep for one engine.

    Args:
        engine_number (int): engine number.
        functions_list (lst): list of keras K.functions.
    """
    eng10 = gen_data_wrapper(X_test_pre, 30, sensors, unit_nrs=np.array([engine_number]))

    ruls = []
    for i in range(len(eng10), 0, -1):
      if i >= 125:
        ruls.append(125)
      else:
        ruls.append(i)

    mus = []
    sigmas = []
    n_predictors = len(functions_list)
    preds = np.zeros(eng10.shape[0]).reshape(-1)

    for p in functions_list:
      pred = p(eng10)[0].mean(axis=1)
      preds += pred.reshape(-1)
    preds = preds/n_predictors


    sigmu = np.zeros(eng10.shape[0]).reshape(-1)
    for p in functions_list:
      sigmas = p(eng10)[1].mean(axis=1)
      mus = p(eng10)[0].mean(axis=1)
      sigmu += (sigmas + mus**2)

    var = sigmu/n_predictors - preds**2
    std = np.sqrt(var)

    plt.plot(preds, label='prediction - mean')
    plt.plot((preds + std), label='prediction + 1std')
    plt.plot((preds - std), label='prediction - 1std')
    plt.legend(loc="lower left")

def get_predictions_ensemble(function_list, x_test, y_test, model_name,clipRUL):
    """Makes prediction for the ensemble.

    Args:
        function_list (lst): list keras K.functions.
        x_test (array-like): testing input samples
        y_test (array-like): ground truth labels
        model_name (str): name of model to be saved
        clipRUL (int): establish upper limit of RUL.

    Returns:
        tuple: average mu and average sigma.
    """
    n_predictors = len(function_list)
    preds = np.zeros(x_test.shape[0]).reshape(-1)
    for p in function_list:
        pred = p(x_test)[0].mean(axis=1)
        preds += pred
    preds = preds/n_predictors

    sigmu = np.zeros(x_test.shape[0]).reshape(-1)
    for p in function_list:
      sigmas = p(x_test)[1].mean(axis=1)
      mus = p(x_test)[0].mean(axis=1)
      sigmu += (sigmas + mus**2)

    var = sigmu/n_predictors - preds**2
    std = np.sqrt(var)
    if clipRUL==True:
      y_test_c = y_test.copy()
      y_test_c['RUL'].clip(upper=125, inplace=True)
      y_test_n = y_test.copy()
    else:
      y_test_c = y_test.copy()
    plt.plot(y_test_c, label='true RUL', c='#91c0cf', linewidth=3.0)
    plt.plot(preds, label='prediction', c = '#2e3d42', )
    plt.legend(loc="upper left")
    preds =  mus

    sum_sq = sum((preds.flatten() - y_test_c.values.flatten())**2)
    sum_sq1 = sum((preds.flatten() - y_test_n.values.flatten())**2)
    print(f"RMSE with clipped RUL is {(sum_sq/100)**(1/2)}")
    print(f"RMSE with clipped RUL is {(sum_sq1/100)**(1/2)}")
    plt.savefig(f'/content/drive/MyDrive/images/_{model_name}_unsorted.png')

    return preds, sigmu

def get_predictions_ensemble1(predictions, x_test, y_test, model_name, clipRUL=True):
    n_predictors = len(predictions)
    preds = np.zeros(x_test.shape[0]).reshape(-1)
    for p in predictions:
        pred = p[0].mean(axis=1)
        preds += pred
    preds = preds/n_predictors

    sigmu = np.zeros(x_test.shape[0]).reshape(-1)
    for p in predictions:
      sigmas = p[1].mean(axis=1)
      mus = p[0].mean(axis=1)
      sigmu += (sigmas + mus**2)

    var = sigmu/n_predictors - preds**2
    std = np.sqrt(var)

    if clipRUL==True:
      y_test_c = y_test.copy()
      y_test_c['RUL'].clip(upper=125, inplace=True)
      y_test_n = y_test.copy()
    else:
      y_test_c = y_test.copy()
    plt.plot(y_test_c, label='true RUL', c='#91c0cf', linewidth=3.0)
    plt.plot(preds, label='prediction', c = '#2e3d42', )

    plt.legend(loc="upper left")
    preds =  mus

    sum_sq = sum((preds.flatten() - y_test_c.values.flatten())**2)
    sum_sq1 = sum((preds.flatten() - y_test_n.values.flatten())**2)
    print(f"RMSE with clipped RUL is {(sum_sq/100)**(1/2)}")
    print(f"RMSE with clipped RUL is {(sum_sq1/100)**(1/2)}")
    plt.savefig(f'/content/drive/MyDrive/images/_{model_name}_unsorted.png')

    return preds, sigmu




def plot_sorted(predictions, y_test, model_name, clipRUL=True):
    """sort RUL in ascending order with its corresponding prediction and the makes plots.

    Args:
        predictions (lst): prediction for RUL
        y_test (array-like): ground truth labels.
        model_name (str): name of the model to be saved
        clipRUL (bool, optional): upper limit for RUL. Defaults to True.
    """
    RUL_vals = y_test.RemainingUsefulLife.sort_values().values
    RUL_idx = y_test.RemainingUsefulLife.sort_values().index
    sorted_preds = []
    for n in RUL_idx:
      sorted_preds.append(predictions[n])

    if clipRUL:
      rul_vals = []
      for val in RUL_vals:
        if val>=125:
          rul_vals.append(125)
        else: rul_vals.append(val)

    else:
      rul_vals = RUL_vals

    plt.plot(rul_vals,  label='true RUL', c='#91c0cf', linewidth=3.0)
    plt.plot(sorted_preds, label='prediction',c = '#2e3d42',)
    plt.legend(loc="upper left")
    plt.savefig(f'/content/drive/MyDrive/images/_{model_name}.png')



def rul_col_series(df):
    """Adds RUL column to the dataframe.

    Args:
        df (df): pandas dataframe with cmapss data.
    """
    max_cycles = df[['engine_no', 'cycles']].groupby('engine_no').max().values
    max_cycles = [n[0] for n in max_cycles]
    rul_column = []
    for c in max_cycles:
        c_rul = list(range(c, 0, -1))
        for r in c_rul:
            rul_column.append(r -1)
    df['RUL'] = rul_column

def add_operating_condition(df):
    """Find the 6 different operating condition combinations and adds a column with them.

    Args:
        df (df): pandas dataframe with cmapss data from dataset 2 or 4.

    Returns:
        df: pandas dataframe with the added column operating condition combination.
    """
    df_op_cond = df.copy()

    df_op_cond['opc1'] = abs(df_op_cond['opc1'].round())
    df_op_cond['opc2'] = abs(df_op_cond['opc2'].round(decimals=2))

    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['opc1'].astype(str) + '_' + \
                        df_op_cond['opc2'].astype(str) + '_' + \
                        df_op_cond['opc3'].astype(str)
    return df_op_cond

def add_health_index(df):
    """Gives the cmapss dataset health index label based upon RUL.

    Args:
        df (df): pandas dataframe with cmapss dataset.

    Returns:
        df: pandas dataframe with the added column class label for health index.
    """
    df_health_index = df.copy()
    df_health_index['health'] = 3
    df_health_index.loc[df_health_index['RUL'] <= 200, 'health'] = 2
    df_health_index.loc[df_health_index['RUL'] <= 125, 'health'] = 1
    df_health_index.loc[df_health_index['RUL'] <= 50, 'health'] = 0
    return df_health_index

def condition_scaler(df_train, df_test, sensor_names):
    """Standarize subset of dataset based on operating condition.

    Args:
        df_train (df): pandas dataframe of training data.
        df_test (df): pandas dataframe of test data.
        sensor_names (_type_): sensor to be used for standarization.

    Returns:
        tuple: dataframe of training and test data standarized.
    """
    # apply operating condition specific scaling

    for condition in df_train['op_cond'].unique():
        scaler = StandardScaler()
        scaler.fit(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_train.loc[df_train['op_cond']==condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_test.loc[df_test['op_cond']==condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond']==condition, sensor_names])
    return df_train, df_test


def exponential_smoothing(df, sensors, n_samples, alpha=0.4):  #  df = df.copy()
    # first, take the exponential weighted mean
    df[sensors] = df.groupby('engine_no')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)

    # second, drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result
    mask = df.groupby('engine_no')['engine_no'].transform(create_mask, samples=n_samples).astype(bool)
    df = df[mask]
    return df

# alpha = 0.1
# # exponential smoothing
# X_train_pre= exponential_smoothing(X_train_pre, sensors, 0, alpha)
# X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)


def gen_train_data(df, sequence_length, columns):
    """Makes cmapss dataset with a sliding window.

    Args:
        df (df): pandas dataframe cmapss dataset.
        sequence_length (int): length of sliding window
        columns (lst): columns to use in dataset.

    Yields:
        array: array of dataset with sliding.
    """
    data = df[columns].values
    num_elements = data.shape[0]

    # -1 and +1 because of Python indexing
    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data[start:stop, :]



def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # -1 because I want to predict the rul of that last row in the sequence, not the next row
    return data_matrix[sequence_length-1:num_elements, :]



def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value) # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:,:] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values

    # specifically yield the last possible sequence
    stop = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :]



def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['engine_no'].unique()

    data_gen = (list(gen_train_data(df[df['engine_no']==unit_nr], sequence_length, columns))
               for unit_nr in unit_nrs)
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    return data_array

def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['engine_no'].unique()

    label_gen = [gen_labels(df[df['engine_no']==unit_nr], sequence_length, label)
                for unit_nr in unit_nrs]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return label_array



def process_data_standard(dataset, sensors, data_folder, algo='regr', clust=0, include_settings=False):
    train_file = 'train_'+dataset+'.txt'
    test_file = 'test_'+dataset+'.txt'
    # columns
    index_names = ["engine_no", "cycles"]
    setting_names = ["opc1", "opc2", "opc3"]

    col_names = index_names + setting_names + sensor_names

    train = pd.read_csv((data_folder+train_file), sep=r'\s+', header=None,
              names=col_names)
    test = pd.read_csv((data_folder+test_file), sep=r'\s+', header=None,
              names=col_names)
    y_test = pd.read_csv((data_folder+'RUL_'+dataset+'.txt'), sep=r'\s+', header=None,
              names=['RUL'])


    # create RUL values according to the piece-wise target function
    rul_col_series(train)
    if algo == 'regr':
        train['RUL'].clip(upper=125, inplace=True)
    train = add_health_index(train)

    if include_settings:
      sensors = sensors + setting_names
    # remove unused sensors
    drop_sensors = [element for element in sensor_names if element not in sensors]

    # scale with respect to the operating condition
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))



    X_train_pre, X_test_pre = condition_scaler(X_train_pre, X_test_pre, sensors)



    # if dataset in ('FD002', 'FD004'):
    #     x_gmm = X_train_pre.copy()
    #     gmm = GaussianMixture(n_components = 5)

    #     # Fit the GMM model for the dataset
    #     # which expresses the dataset as a
    #     # mixture of 3 Gaussian Distribution
    #     gmm.fit(x_gmm[sensors])
    #     X_train_pre['cluster'] = gmm.predict(x_gmm[sensors])
    #     X_train_pre_clust1 = X_train_pre.loc[X_train_pre.cluster==1]
    #     X_train_pre_clust0 = X_train_pre.loc[X_train_pre.cluster==0]

    #     if clust == 0:
    #       X_train_pre = X_train_pre_clust0
    #     else: X_train_pre = X_train_pre_clust1



    sequence_length = 30


    # train-val split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
    # generate the train/val for *each* sample -> for that we iterate over the train and val units we want
    # this is a for that iterates only once and in that iterations at the same time iterates over all the values we want,
    # i.e. train_unit and val_unit are not a single value but a set of training/vali units

    for train_unit, val_unit in gss.split(X_train_pre['engine_no'].unique(), groups=X_train_pre['engine_no'].unique()):
        train_unit = X_train_pre['engine_no'].unique()[train_unit]  # gss returns indexes and index starts at 1
        val_unit = X_train_pre['engine_no'].unique()[val_unit]


        x_train = gen_data_wrapper(X_train_pre, sequence_length, sensors, train_unit)
        y_train = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], train_unit)

        if algo != 'regr':
            y_train = gen_label_wrapper(X_train_pre, sequence_length, ['health'], train_unit)


        x_val = gen_data_wrapper(X_train_pre, sequence_length, sensors, val_unit)
        y_val = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], val_unit)


        if algo != 'regr':
            y_val = gen_label_wrapper(X_train_pre, sequence_length, ['health'], val_unit)

    # create sequences for test
    test_gen = (list(gen_test_data(X_test_pre[X_test_pre['engine_no']==unit_nr], sequence_length, sensors, -99.))
            for unit_nr in X_test_pre['engine_no'].unique())
    x_test = np.concatenate(list(test_gen)).astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test, X_train_pre, X_test_pre

def process_data_standard2(dataset, sensors, data_folder, condition, sequence_length = 30, algo='regr', clust=-1, get_by_cond=False,  include_settings=False):
    """preprocess the data and makes it ready for training, validation and testing.

    Args:
        dataset (str): name of dataset
        sensors (lst): sensors to use
        data_folder (str): path to folder of cmapss data
        condition (str): operating condition to make a subset of
        sequence_length (int, optional): sliding window length. Defaults to 30.
        algo (str, optional): alorithm to use for datset, regression or classification. Defaults to 'regr'.
        clust (int, optional): if you want to cluster on failure mode. Defaults to -1.
        get_by_cond (bool, optional): if wants to make a subset based on operating conditions. Defaults to False.
        include_settings (bool, optional): if want to use operating conditions features in dataset. Defaults to False.

    Returns:
        tuple: training datasets, test datasets and validation datasets
    """
    train_file = 'train_'+dataset+'.txt'
    test_file = 'test_'+dataset+'.txt'
    # columns
    index_names = ["engine_no", "cycles"]
    setting_names = ["opc1", "opc2", "opc3"]

    col_names = index_names + setting_names + sensor_names

    train = pd.read_csv((data_folder+train_file), sep=r'\s+', header=None,
              names=col_names)
    test = pd.read_csv((data_folder+test_file), sep=r'\s+', header=None,
              names=col_names)
    y_test = pd.read_csv((data_folder+'RUL_'+dataset+'.txt'), sep=r'\s+', header=None,
              names=['RUL'])


    # create RUL values according to the piece-wise target function
    rul_col_series(train)
    if algo == 'regr':
        train['RUL'].clip(upper=125, inplace=True)
    train = add_health_index(train)

    # if include_settings is True, then includes the operating conditions
    if include_settings:
      sensors = sensors + setting_names
    # remove unused sensors
    drop_sensors = [element for element in sensor_names if element not in sensors]

    # scale with respect to the operating condition
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))



    X_train_pre, X_test_pre = condition_scaler(X_train_pre, X_test_pre, sensors)


    # If clust is 0 or 1, performs GMM clustering and returns the requested cluster to continue processing
    if clust in [0,1]:
        sequence_length = sequence_length
        x_gmm = X_train_pre.copy()
        gmm = GaussianMixture(n_components = 2)

        # Fit the GMM model for the dataset
        # which expresses the dataset as a
        # mixture of 3 Gaussian Distribution
        gmm.fit(x_gmm[sensors])
        X_train_pre['cluster'] = gmm.predict(x_gmm[sensors])
        X_train_pre_clust1 = X_train_pre.loc[X_train_pre.cluster==1]
        X_train_pre_clust0 = X_train_pre.loc[X_train_pre.cluster==0]

        if clust == 0:
          X_train_pre = X_train_pre_clust0
        else: X_train_pre = X_train_pre_clust1

    # if get_by_cond is True, returns a processed subset of the requested operating condition
    if get_by_cond:
        X_train_pre = X_train_pre.loc[X_train_pre.op_cond ==condition].reindex()




    # train-val split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
    # generate the train/val for *each* sample -> for that we iterate over the train and val units we want
    # this is a for that iterates only once and in that iterations at the same time iterates over all the values we want,
    # i.e. train_unit and val_unit are not a single value but a set of training/vali units

    for train_unit, val_unit in gss.split(X_train_pre['engine_no'].unique(), groups=X_train_pre['engine_no'].unique()):

        train_unit = X_train_pre['engine_no'].unique()[train_unit]  # gss returns indexes and index starts at 1
        val_unit = X_train_pre['engine_no'].unique()[val_unit]

        x_train = gen_data_wrapper(X_train_pre, sequence_length, sensors, train_unit)
        y_train = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], train_unit)
        # makes health indiex if the process is for algorithms other than regression
        if algo != 'regr':
            y_train = gen_label_wrapper(X_train_pre, sequence_length, ['health'], train_unit)


        x_val = gen_data_wrapper(X_train_pre, sequence_length, sensors, val_unit)
        y_val = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], val_unit)

        # makes health indiex if the process is for algorithms other than regression
        if algo != 'regr':
            y_val = gen_label_wrapper(X_train_pre, sequence_length, ['health'], val_unit)

    # create sequences for test

    test_gen = (list(gen_test_data(X_test_pre[X_test_pre['engine_no']==unit_nr],
                                   sequence_length, sensors, -99.))
            for unit_nr in X_test_pre['engine_no'].unique())
    x_test = np.concatenate(list(test_gen)).astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test, X_train_pre, X_test_pre

def classif_rf(X_train, y_train, X_test, y_test, n_estimators=10, max_depth=10, random_state=237,cv=10):
    """Train a RF classifier. Calculate the accuracy and apply crossvalidation.

    Args:
        X_train ({array-like, sparse matrix}): training input samples.
        y_train (1-d array-like): training label targets
        X_test ({array-like, sparse matrix}): testing input samples.
        y_test (1-d array-like): ground truth labels
        n_estimators (int, optional): number of trees in the forest. Defaults to 10.
        max_depth (int, optional): maximum depth of the tree. Defaults to 10.
        random_state (int, optional): control the randomness. Defaults to 237.
        cv (int, optional): Specify number of folds in crossvalidation. Defaults to 10.

    Returns:
        tuple : random forest fitted model, y_test and the predicted label y_pred.
    """
    # instantiate the classifier
    rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,  random_state=random_state)
    X_train = [x[-1] for x in X_train]
    X_test = [x[-1] for x in X_test]
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    # fit the model
    rfc.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = rfc.predict(X_test)

    # Check accuracy score
    print(f'Model accuracy score with 10 decision-trees : {accuracy_score(y_test, y_pred):.4f}')

    # Apply crossvalidation.
    scores_rfc = cross_val_score(rfc, X_test, y_test, cv=cv)
    print(f"\nThe scores for cross validation are:\n {scores_rfc}")
    print(f"\nThe mean score is: {scores_rfc.mean():.4f}\n")

    #Building classification report
    print(classification_report(y_test,y_pred))

    return rfc, y_test, y_pred

def parameter_tuning(clf, param_grid, X_train, y_train, X_test, y_test, cv):
    # Initialize the model
    X_train = [x[-1] for x in X_train]
    X_test = [x[-1] for x in X_test]
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    clf = clf
    # Initialize gridsearchcv.
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, n_jobs=-1)
    # Fit the model to the data.
    grid_search.fit(X_train, y_train)
    # Get the best parameters and best estimators.
    best_params = grid_search.best_params_
    best_estim = grid_search.best_estimator_
    best_score = grid_search.best_score_
    # Make predictions.
    y_pred = best_estim.predict(X_test)

    # Evaluate the model.
    accuracy_grid = accuracy_score(y_test, y_pred)
    #print(f'Accuracy: {best_score:.3f}')
    print(f'Accuracy: {accuracy_grid:.3f}')

    return grid_search

def classif_svm(X_train, y_train, X_test, y_test, random_state=237,cv=10):
    """Train a SVM classifier. Calculate the accuracy and apply crossvalidation.

    Args:
        X_train ({array-like, sparse matrix}): training input samples.
        y_train (1-d array-like): training label targets
        X_test ({array-like, sparse matrix}): testing input samples.
        y_test (1-d array-like): ground truth labels
        random_state (int, optional): control the randomness. Defaults to 237.
        cv (int, optional): Specify number of folds in crossvalidation. Defaults to 10.

    Returns:
        tuple : svm fitted model, y_test and the predicted label y_pred.
    """
    # instantiate classifier with default hyperparameters
    svc=SVC(random_state=random_state)
    X_train = [x[-1] for x in X_train]
    X_test = [x[-1] for x in X_test]
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    # fit classifier to training set
    svc.fit(X_train,y_train)
    # make predictions on test set
    y_pred = svc.predict(X_test)
    # compute and print accuracy score
    print(f'Model accuracy score with default hyperparameters: {accuracy_score(y_test, y_pred):0.4f}')

    # Apply crossvalidation.
    scores_svc = cross_val_score(svc, X_test, y_test, cv=cv)
    print(f"\nThe scores for cross validation are:\n {scores_svc}")
    print(f"\nThe mean score is: {scores_svc.mean():.4f}")

    #Building classification report
    print(classification_report(y_test,y_pred,  zero_division=0))

    return svc, y_test, y_pred

def classif_lda(X_train, y_train, X_test, y_test,cv=10):
    """Train a LDA classifier. Calculate the accuracy and apply crossvalidation.

        Args:
            X_train ({array-like, sparse matrix}): training input samples.
            y_train (1-d array-like): training label targets
            X_test ({array-like, sparse matrix}): testing input samples.
            y_test (1-d array-like): ground truth labels
            cv (int, optional): Specify the number of folds in crossvalidation.
            Defaults to 10.

        Returns:
            tuple : lda fitted model, y_test and the predicted label y_pred.
    """
    # instantiate the classifier
    lda = LinearDiscriminantAnalysis()

    # Rearrange the data to make it suitable.
    X_train = [x[-1] for x in X_train]
    X_test = [x[-1] for x in X_test]
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # fit the model
    lda.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = lda.predict(X_test)

    # Check accuracy score
    print(f'Model accuracy score with default parameters : {accuracy_score(y_test, y_pred):.4f}')

    # Apply crossvalidation.
    scores = cross_val_score(lda, X_test, y_test, cv=cv)
    print(f"\nThe scores for cross validation are:\n {scores}")
    print(f"\nThe mean score is: {scores.mean():.4f}\n")

    # Building classification report
    print(classification_report(y_test,y_pred, zero_division=0))

    return lda, y_test, y_pred

def train_elman(model_name, x_train, y_train, x_val, y_val, window_length, neurons1, neurons2= None, optimizer='Adadelta'):
    """Creates the model for a elman network with specified parameters.

    Args:
        model_name (str): name of the model to be saved
        x_train ({array-like, sparse matrix}): training input samples
        y_train (array-like): training target values
        x_val ({array-like, sparse matrix}): validation input samples
        y_val (array-like): validation target values
        window_length (int): length of sliding window
        neurons1 (int): number of neurons in hidden layer
        neurons2 (int, optional): number of neurons in 2. hidden layer if use 2 hidden layers. Defaults to None.
        optimizer (str, optional): optimizer of the network. Defaults to 'Adadelta'.

    Returns:
        tuple: model1_1 - fitted model of elman network
               history1_1 - history of training and validation of the model.
    """
    callback =tf.keras.callbacks.EarlyStopping(monitor='val_acc', start_from_epoch=35, patience=20)
    checkpoint_path = f"/content/drive/MyDrive/classiff_results/elman_models/"+model_name+".ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    #cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(0.01, 500, alpha=0.0, name=None)

    optimizer=optimizer
    #optimizer = tf.keras.optimizers.Adadelta(learning_rate=cosine_decay_scheduler)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=0, monitor='val_acc' ,save_best_only=True)
    inputs = Input(shape=(window_length, len(sensors)))
    x = SimpleRNN(neurons1)(inputs)
    if neurons2 is not None:
      x = SimpleRNN(neurons2)(x)
    x = Dense(4, activation='softmax')(x)
    start = time.time()

    model1_1 = Model(inputs, x)
    model1_1.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    print(model1_1.summary())
    history1_1 = model1_1.fit(x_train, y_train,
                        epochs=200, validation_data = (x_val, y_val), callbacks=[cp_callback, callback],
                        verbose=2)

    print(time.time()-start)
    return model1_1, history1_1

def create_stacking_dataset(model_lda, model_rf, model_svm, model_elman, X_test):
    """creates a stacking dataset by combining predicted values from multiple
    classifiers.

    Args:
        model_lda (LinearDiscriminantAnalysis): trained LDA classifier.
        model_rf (RandomForestClassifier): trained Random Forest classifier
        model_svm (SVC): trained SVM classifier
        model_elman (_type_): trained elman net classifier
        X_test ({array-like, sparse matrix}): testing input samples.

    Returns:
        2-d array: array formed by stacking y_pred of the different classifiers.
    """
    X_test_fix = [x[-1] for x in X_test]
    # Predicting
    # lda.
    y_pred_lda = model_lda.predict(X_test_fix)
    # random forest
    y_pred_forest = model_rf.predict(X_test_fix)
    # svm
    y_pred_svm = model_svm.predict(X_test_fix)
    # elman
    y_pred_elman = model_elman.predict(X_test).argmax(axis=1)
    # Create stacking dataset.
    stacking = np.column_stack((y_pred_lda, y_pred_forest, y_pred_svm, y_pred_elman))
    return stacking

def stacking(X_train, y_train, X_test, y_test, n_estimators=20, max_depth=5, random_state=237, cv=10):
    """Trains a stacked gradient boosting classifier.

    Args:
        X_train ({array-like, sparse matrix}): training input samples
        y_train (array-like): training target values
        X_test ({array-like, sparse matrix}): testing input samples.
        y_test (1d array-like): ground truth labels.
        n_estimators (int, optional): number of boosting stages to perform. Defaults to 20.
        max_depth (int, optional): maximum depth of the individual estimators. Defaults to 5.
        random_state (int, optional): controls the randomness. Defaults to 237.
        cv (int, optional): number of folds for crossvalidation. Defaults to 10.

    Returns:
        tuple(stack_boost,y_test,y_pred): stack_boost - trained GradientBoostingClassifier
                                           y_test      - ground truth labels.
                                           y_pred      - predicted labels returned by classifier
    """
    y_train = y_train.flatten()
    #y_test = y_test.flatten()

    # add health index to y.
    y_test = add_health_index(y_test)
    # set y_test to be label of health index.
    y_test = y_test.health.values

    # Instantiate Stacking classifier
    stack_boost = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # fit the model.
    stack_boost.fit(X_train, y_train)

    # Predict the test set results.
    y_pred = stack_boost.predict(X_test)

    # Check accuracy score
    print(f'Model accuracy score : {accuracy_score(y_test, y_pred):.4f}')

    # Apply crossvalidation.
    scores_stack = cross_val_score(stack_boost, X_test, y_test, cv=cv)
    print(f'\nThe scores for cross validation of gradient boosting are:\n {scores_stack}')
    print(f'\nThe mean score of gradient boosting is: {scores_stack.mean()}\n')

    #Building classification report
    print(classification_report(y_test,y_pred))

    return stack_boost, y_test, y_pred

def train_lda(solver, shrinkage, X_train, y_train, cv):
    """Trains a Linear Disciminant Analysis Classifier.

    Args:
        solver ({‘svd’, ‘lsqr’, ‘eigen’}): Solver to use.
        shrinkage (‘auto’ or float): Shrinkage parameter.
        X_train ({array-like, sparse matrix}): training input samples
        y_train (array-like): target values
        cv (int, optional): Number of folds for crossvalidation

    Returns:
        LinearDiscriminantAnalysis: trained LDA classifier.
    """
    # instantiate the classifier
    lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    # Rearrange the data to make it suitable.
    X_train = [x[-1] for x in X_train]
    y_train = y_train.flatten()

    # fit the model
    lda.fit(X_train, y_train)
    return lda

def train_svm(C, gamma, kernel, X_train, y_train, random_state=237,cv=10):
    """Trains a Support Vector Machine Classifier.

    Args:
        C (float): Regularization parameter.
        gamma ({‘scale’, ‘auto’} or float): Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        kernel ({‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable): Specifies the kernel type to be used.
        X_train ({array-like, sparse matrix}): training input samples
        y_train (array-like): target values
        random_state (int, optional): Controls the randomness. Defaults to 237.
        cv (int, optional): Number of folds for crossvalidation. Defaults to 10.

    Returns:
        SVC: trained SVM classifier.
    """
    # instantiate classifier
    svc=SVC(C=C, gamma=gamma, kernel=kernel, random_state=random_state)
    X_train = [x[-1] for x in X_train]
    y_train = y_train.flatten()
    # fit classifier to training set
    svc.fit(X_train,y_train)
    return svc

def train_rf(max_depth, max_features, min_samples_split, n_estimators, X_train, y_train, random_state=237,cv=10):
    """Trains a Random Forest Classifier.

    Args:
        max_depth (int): The maximum depth of the tree.
        max_features ({“sqrt”, “log2”, None}, int or float): number of features when looking for the best split
        min_samples_split (int or float): minimum number of samples to split an internal node
        n_estimators (int): number of trees in the forest
        X_train ({array-like, sparse matrix}): training input samples
        y_train (array-like): target values
        random_state (int, optional): Controls the randomness. Defaults to 237.
        cv (int, optional): Number of folds for crossvalidation. Defaults to 10.

    Returns:
        RandomForestClassifier: trained RandomForestClassifier
    """
    # instantiate the classifier
    rfc = RandomForestClassifier(max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, n_estimators=n_estimators, random_state=random_state)
    # Rearrange the data.
    X_train = [x[-1] for x in X_train]
    y_train = y_train.flatten()
    # fit the model
    rfc.fit(X_train, y_train)
    return rfc