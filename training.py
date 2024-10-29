from matplotlib.pylab import randint
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_models(file_path='data/powerconsumption.csv'):
    data = pd.read_csv(file_path)
    X = data[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']]
    Y = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_dev, X_test, Y_dev, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42)
    }

    param_distributions = {
        'LinearRegression': {
            'fit_intercept': [True, False],
            'copy_X': [True, False],
            'n_jobs': [None, -1],
            'positive': [True, False]
        },
        'DecisionTree': {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10],
            'max_features': ['sqrt', 'log2', None]
        },
        'RandomForest': {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
    }

    tuned_models = {}

    for model_name, model in models.items():
        print(f"Configure parameters for {model_name}")
    
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[model_name],
            n_iter=10,
            scoring='neg_mean_absolute_error',
            cv=3,
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_dev, Y_dev)
        tuned_models[model_name] = search.best_estimator_

        print(f"Best parameters for {model_name}: {search.best_params_}")
        print(f"Best MAE for {model_name}: {-search.best_score_:.2f}")


    for model_name, model in tuned_models.items():
        model.fit(X_train, Y_train)
        print(f"{model_name} model trained.")


    return tuned_models, X_test, Y_test


if __name__ == '__main__':
    train_models()
