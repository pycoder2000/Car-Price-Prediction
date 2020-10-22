import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import tree

import config

if __name__ == "__main__":
   df = pd.read_csv(config.TRAINING_FILE_ENCODED)

   X = df.drop(['Selling_Price', 'id'], 1).values
   y = df.Selling_Price.values

   regressor = tree.DecisionTreeRegressor()

   param_grid = {
      'splitter' : ["best", "random"],
      "max_depth": np.arange(1, 31),
      "criterion": ["mse", "friedman_mse", "mae"]
   }

   model = model_selection.RandomizedSearchCV(
      estimator=regressor,
      param_distributions=param_grid,
      n_iter=1000,
      verbose=10,
      n_jobs=1,
      cv=5
   )

   model.fit(X, y)
   print(f"best score: {model.best_score_}")

   print("best parameter set: ")
   best_param = model.best_estimator_.get_params()

   for param_name in sorted(param_grid.keys()):
      print(f"\t{param_name} : {best_param[param_name]}")



"""
n_iter = 20
---------------------------------------

best score: 0.92980662507159
best parameter set:
        criterion : mse
        max_depth : 15
        splitter  : best
---------------------------------------

********************************************************************

n_iter = 100
---------------------------------------
best score: 0.9441216438960449
best parameter set:
        criterion : friedman_mse
        max_depth : 15
        splitter : best
---------------------------------------


---------------------------------------
---------------------------------------
---------------------------------------
---------------------------------------
best score: 0.8920043321779605
best parameter set:
        criterion : mae
        max_depth : 18
        splitter : best
"""

