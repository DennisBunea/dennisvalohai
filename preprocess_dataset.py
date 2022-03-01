import valohai
import tensorflow as tf
import csv
from csv import reader
import sklearn
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print('Loading data')
data_train = pd.read_csv(valohai.inputs("train").path())
data_test = pd.read_csv(valohai.inputs("test").path())
valohai.prepare(
      step='preprocess-dataset',
      image='python:3.9')

default_inputs = {
    'train': 'datum://017ef88d-2343-ef70-a47c-1ed37b59b244',
        'test': 'datum://017ef88d-21b8-2413-6212-714e6dd770a8',
            'gender_submission': 'datum://017ef88d-2036-4ea5-7755-9d1d303548cf',

   
}



default_parameters = {'n_estimators': 4,
              'max_features': 'log2', 
              'criterion': 'entropy',
              'max_depth': 2 , 
              'min_samples_split': 2 ,
              'min_samples_leaf': 1
             }
print('Preprocessing data')
valohai.prepare(step="train", image="tensorflow/tensorflow:2.6.1-gpu", default_inputs=default_inputs , default_parameters=default_parameters)
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

data_train = transform_features(data_train)
data_test = transform_features(data_test)

from sklearn import preprocessing, utils
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(data_train, data_test)
print('Saving preprocessed data')




#Drazen if you are taking a look I know it`s incomplete I`ll fix that till Monday! 
#PS: I was worked at tutorial.
