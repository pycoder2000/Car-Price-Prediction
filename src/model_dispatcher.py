from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
import xgboost as xgb

models = {
   "decision_tree_gini" : tree.DecisionTreeRegressor(
      criterion='gini'
      ),
   "decision_tree_entropy" : tree.DecisionTreeRegressor(
      criterion="entropy"
      ),
   "rf" : ensemble.RandomForestRegressor(),
   "Linres": linear_model.LinearRegression(),
   "xgb_rf_reg": xgb.XGBRFRegressor(),
   "xgb_reg": xgb.XGBRegressor()
}