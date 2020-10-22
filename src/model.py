import pandas as pd
import numpy as np
import os

import config
import joblib

from sklearn import model_selection
from sklearn import tree
from sklearn import metrics

def train():
   data = pd.read_csv(config.TRAINING_FILE_ENCODED)

   X = data.drop(['Selling_Price', 'id', 'kfold'], 1).values
   y = data.Selling_Price.values

   model = tree.DecisionTreeRegressor(
      criterion = "mae",
      max_depth = 18,
      splitter = "best"
   )

   model.fit(X, y)

   os.makedirs(os.path.join(config.MODEL_OUTPUT, "DecisionTree/"))

   joblib.dump(
      model,
      os.path.join(config.MODEL_OUTPUT, "DecisionTree/model.pkl")
   )

if __name__=="__main__":
   train()