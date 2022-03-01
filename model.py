import csv
import valohai
from csv import reader
import pandas as pd
data_train = pd.read_csv(valohai.inputs("train").path())
data_test = pd.read_csv(valohai.inputs("test").path())
from sklearn.model_selection import train_test_split
X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
#from sklearn.model_selection import GridSearchCV
# Choose the type of classifier.
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=2, max_features='log2', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=4, n_jobs=1,
            oob_score=False, random_state=23, verbose=0,
            warm_start=False)
acc_scorer = make_scorer(accuracy_score)

#grid_obj = GridSearchCV(clf, default_parameters, scoring=acc_scorer)
#grid_obj = grid_obj.fit(X_train, y_train)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
#clf = grid_obj.best_estimator_

# Run the grid search
#grid_obj = GridSearchCV(clf, default_parameters, scoring=acc_scorer)
#grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
#clf = grid_obj.best_estimator_