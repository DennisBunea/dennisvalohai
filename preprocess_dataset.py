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

  
print('Preprocessing data')

  
print('Saving preprocessed data')
path = valohai.outputs().path('train','test')


#Drazen if you are taking a look I know it`s incomplete I`ll fix that till Monday! 
#PS: I was worked at tutorial.
