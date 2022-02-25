import numpy as np
import valohai
import csv
from csv import reader
import pandas as pd
data_train = pd.read_csv(valohai.inputs("train").path())
data_test = pd.read_csv(valohai.inputs("test").path())
valohai.prepare(
      step='preprocess-dataset',
      image='python:3.9',
      default_inputs={
          'train': 'datum://017ef88d-2343-ef70-a47c-1ed37b59b244',
          'test': 'datum://017ef88d-21b8-2413-6212-714e6dd770a8',
          'gender_submission': 'datum://017ef88d-2036-4ea5-7755-9d1d303548cf',
      },
  )

print('Loading data')
with open(valohai.inputs("train","test").path()) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    x_train, y_train = csv_file('x_train'), csv_file('y_train')
    x_test, y_test = csv_file('x_test'), csv_file('y_test')
  
print('Preprocessing data')
x_train, x_test = x_train / 255.0, x_test / 255.0
  
print('Saving preprocessed data')
path = valohai.outputs().path('train','test')
np.savez_compressed(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
