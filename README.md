# Car Price Prediction

An app made using Flask to predict the car price using details like year of purchase, purchasing price, number of owners, etc.

Technique used: 
  - Decision Tree Regressor
  - Random Forest Regressor
---
## Information

The template used can be found on [index.html](https://github.com/pycoder2000/Car-Price-Prediction/blob/main/templates/index.html)  

The dataset used here is very small and can be found at [car data.csv](https://github.com/pycoder2000/Car-Price-Prediction/blob/main/input/car%20data.csv)

---
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
python -m pip install -U pip setuptools
pip install -U -r requirements.txt
```

---
## How to run :

1. Run [create_folds.py](https://github.com/pycoder2000/Car-Price-Prediction/blob/main/src/create_folds.py) from src to create folds in the dataset.
2. Run [encode_data.py](https://github.com/pycoder2000/Car-Price-Prediction/blob/main/src/encode_data.py) from src to encode the data using pandas dummy method and to select the features.
3. Run [model.py](https://github.com/pycoder2000/Car-Price-Prediction/blob/main/src/model.py) to train the model and dump it using joblib at [model.pkl](https://github.com/pycoder2000/Car-Price-Prediction/blob/main/models/DecisionTree/model.pkl)
    - you can check the [model.pkl](https://github.com/pycoder2000/Car-Price-Prediction/blob/main/models/model.pkl) also, it is trained on random forest regressor and is shown in [EDA.ipynb](https://github.com/pycoder2000/Car-Price-Prediction/blob/main/notebooks/EDA.ipynb)
4. Run [app.py](https://github.com/pycoder2000/Car-Price-Prediction/blob/main/app.py)


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
