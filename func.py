
import pandas as pd
import os


def get_data(hierar = False, oneHot = False):

    #Gets path to folder
    path = os.getcwd()
    train_path = path + '/train_heart.csv'
    test_path = path + '/test_heart.csv'

    #variable for the categorical data columns in the dataset
    col = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
    print(col)
    # Load Data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    if hierar:
        age_train = df_train['age']
        age_test= df_test['age']
        age = [age_train, age_test ]
    else: 
        age = []

    if oneHot:
        # One hot encode data: 
        X_train = pd.get_dummies(df_train,columns=col)
        X_test = pd.get_dummies(df_test,columns=col)

        #Extract y values:
        y_train = X_train['target'].values.astype("int")
        y_test = X_test['target'].values.astype("int")

        #Extract data 
        X_train = X_train.loc[:, X_train.columns != 'target'].values
        X_test = X_test.loc[:, X_test.columns != 'target'].values

        #Normalize non one-hot-encoded data (training data)
        X_train_temp = X_train[:,0:5].astype('float')
        X_mean = X_train_temp.mean(axis=0)
        X_std = X_train_temp.std(axis=0)
        X_train_temp = (X_train_temp - X_mean) / X_std
        X_train[:,0:5] = X_train_temp

        #Normalize non one-hot-encoded data (training data)
        X_train_temp = X_test[:,0:5].astype('float')
        X_mean = X_train_temp.mean(axis=0)
        X_std = X_train_temp.std(axis=0)
        X_train_temp = (X_train_temp - X_mean) / X_std
        X_test[:,0:5] = X_train_temp
    else:
        X_train = df_train
        X_test = df_test

        #Extract y values:
        y_train = X_train['target'].values.astype("int")
        y_test = X_test['target'].values.astype("int")

        #Extract data 
        X_train = X_train.loc[:, X_train.columns != 'target'].values
        X_test = X_test.loc[:, X_test.columns != 'target'].values

        #Normalize non one-hot-encoded data (training data)
        X_train_temp = X_train.astype('float')
        X_mean = X_train_temp.mean(axis=0)
        X_std = X_train_temp.std(axis=0)
        X_train_temp = (X_train_temp - X_mean) / X_std
        X_train = X_train_temp

        #Normalize non one-hot-encoded data (training data)
        X_train_temp = X_test.astype('float')
        X_mean = X_train_temp.mean(axis=0)
        X_std = X_train_temp.std(axis=0)
        X_train_temp = (X_train_temp - X_mean) / X_std
        X_test = X_train_temp
  

    return [X_train,y_train,X_test,y_test,age]

