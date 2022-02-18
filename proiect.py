import valohai
import tensorflow as tf
import csv
from csv import reader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
data_train = pd.read_csv(valohai.inputs('myinput').path())
data_test = pd.read_csv(valohai.inputs('myinput').path())


default_inputs = {
    'myinput': 'datum://017ef88d-2343-ef70-a47c-1ed37b59b244',
   
}

default_parameters = {
    'iterations': 10,
    'epoch': 10,
    'learning_rate': 0.001,

}

def log_metadata(epoch, logs):
     with valohai.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])

input_path = valohai.inputs('myinput').path()
data_train, data_train = ['data_train'], ['data_train']
data_test, data_test = ['data_test'], ['data_test']
data_train, data_test = data_train / 255.0, data_test / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
 
optimizer = tf.keras.optimizers.Adam(learning_rate=valohai.parameters('learning_rate').value)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy'])
callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
model.fit(data_train, data_train, epochs=valohai.parameters('epoch').value, callbacks=[callback])
model.evaluate(data_test,  data_test, verbose=2)
output_path = valohai.outputs().path('model.h5')
model.save(output_path)

valohai.prepare(step="train", image="tensorflow/tensorflow:2.6.1-gpu", default_inputs=default_inputs , default_parameters=default_parameters)

# Open the CSV file from Valohai inputs
with open(valohai.inputs('myinput').path()) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    
for i in range(valohai.parameters('iterations').value):
    print("Iteration %s" % i)

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)
plt.show()


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

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=5, max_features='log2', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=9, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))


out_path = valohai.outputs().path('myinput')
def to_csv(df):
    df.to_csv(out_path)


