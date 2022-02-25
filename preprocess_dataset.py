import numpy as np
import valohai
  
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
with np.load(valohai.inputs('train', 'test').path(), allow_pickle=True) as file:
      x_train, y_train = file['x_train'], file['y_train']
      x_test, y_test = file['x_test'], file['y_test']
  
print('Preprocessing data')
x_train, x_test = x_train / 255.0, x_test / 255.0
  
print('Saving preprocessed data')
path = valohai.outputs().path('train','test')
np.savez_compressed(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
