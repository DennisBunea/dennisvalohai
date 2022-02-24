import valohai
import tensorflow as tf
import csv
from csv import reader
import sklearn
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
data_train = pd.read_csv(valohai.inputs("train").path())
data_test = pd.read_csv(valohai.inputs("test").path())


default_inputs = {
    'train': 'datum://017ef88d-2343-ef70-a47c-1ed37b59b244',
        'test': 'datum://017ef88d-21b8-2413-6212-714e6dd770a8',
            'gender_submission': 'datum://017ef88d-2036-4ea5-7755-9d1d303548cf',

   
}

default_parameters = {'n_estimators': [4],
              'max_features': ['log2'], 
              'criterion': ['entropy'],
              'max_depth': [2] , 
              'min_samples_split': [2] ,
              'min_samples_leaf': [1]
             }

valohai.prepare(step="train", image="tensorflow/tensorflow:2.6.1-gpu", default_inputs=default_inputs , default_parameters=default_parameters)

input_path = valohai.inputs("train","test").path()



# Open the CSV file from Valohai inputs
with open(valohai.inputs("train","test").path()) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)

plt.savefig(valohai.outputs().path("mygraph.png"))


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

from sklearn.model_selection import train_test_split
X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=2, max_features='log2', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=4, n_jobs=1,
            oob_score=False, random_state=23, verbose=0,
            warm_start=False)
acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, default_parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
clf = grid_obj.best_estimator_

print(accuracy_score(y_test, predictions))
with valohai.metadata.logger() as logger:
    logger.log("accuracy", accuracy_score(y_test, predictions))

# Choose the type of classifier. 



# Type of scoring used to compare parameter combinations


# Run the grid search
#grid_obj = GridSearchCV(clf, default_parameters, scoring=acc_scorer)
#grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
#clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 



out_path = valohai.outputs().path("train", "test")
print(out_path)