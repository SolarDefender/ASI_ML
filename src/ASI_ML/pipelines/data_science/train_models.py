import wandb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_models(X_train, Y_train, X_dev, Y_dev, parameters):
    # Initialize W&B run
    wandb.init(
        project=parameters['wandb']['project'],
        entity=parameters['wandb']['entity'],
        name='train_models',
        config=parameters  # Logs all parameters
    )

    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTreeRegressor': DecisionTreeRegressor(random_state=parameters["random_state"]),
        'RandomForestRegressor': RandomForestRegressor(random_state=parameters["random_state"])
    }

    param_distributions = parameters["hyperparameters"]
    n_iter = parameters["n_iter"]
    cv_folds = parameters["cv_folds"]

    tuned_models = {}

    for model_name, model in models.items():
        print(f"Configuring parameters for {model_name}")

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[model_name],
            n_iter=n_iter,
            scoring='neg_mean_absolute_error',
            cv=cv_folds,
            random_state=parameters["random_state"],
            n_jobs=-1
        )

        search.fit(X_dev, Y_dev)
        best_model = search.best_estimator_
        tuned_models[model_name] = best_model

        # Log hyperparameters and metrics to W&B
        wandb.log({
            f"{model_name}_best_params": search.best_params_,
            f"{model_name}_best_score": -search.best_score_
        })

        print(f"Best parameters for {model_name}: {search.best_params_}")
        print(f"Best MAE for {model_name}: {-search.best_score_:.2f}")

    # Train models on the full training data
    for model_name, model in tuned_models.items():
        model.fit(X_train, Y_train)
        print(f"{model_name} model trained.")

    wandb.finish()

    return tuned_models