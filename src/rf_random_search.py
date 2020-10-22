import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import ensemble

import config

if __name__ == "__main__":
   df = pd.read_csv(config.TRAINING_FILE_ENCODED)

   X = df.drop(['Selling_Price', 'id'], 1).values
   y = df.Selling_Price.values

   regressor = ensemble.RandomForestRegressor(n_jobs=-1)

   param_grid = {
      "n_estimators": np.arange(100, 1500, 100),
      "max_depth": np.arange(1, 31),
      "criterion": ["mse", "mae"]
   }

   model = model_selection.RandomizedSearchCV(
      estimator=regressor,
      param_distributions=param_grid,
      n_iter=30,
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
---------------------------------------
n_iter=20
***********


best score: 0.9191538523823983
best parameter set:
        criterion : mse
        max_depth : 18
        n_estimators : 100
---------------------------------------
n_iter=100
***********
best score: 0.9177930850574993
best parameter set:
        criterion : mse
        max_depth : 15
        n_estimators : 100
---------------------------------------
---------------------------------------
---------------------------------------
---------------------------------------
---------------------------------------
n_iter=30
***********

best score: 0.9064792708645513
best parameter set:
        criterion : mse
        max_depth : 12
        n_estimators : 600

"""