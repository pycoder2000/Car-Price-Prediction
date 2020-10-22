import numpy as np
import pandas as pd
from sklearn import preprocessing

import config

def encode():
   data = pd.read_csv(config.TRAINING_FILE)

   data['Current_Year'] = 2020
   data['Number of years'] = data['Current_Year'] - data['Year']

   dataset =  pd.get_dummies(
      data.drop(
         ['Car_Name', 'Current_Year', 'Year'],
         axis = 1),
      drop_first=True)

   dataset.to_csv(config.TRAINING_FILE_ENCODED)

if __name__ == '__main__':
   encode()