import pandas as pd
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(models, X_test, Y_test, parameters):
    # Initialize W&B run
    wandb.init(
        project=parameters['wandb']['project'],
        entity=parameters['wandb']['entity'],
        name='evaluate_models'
    )

    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        mae = mean_absolute_error(Y_test, predictions)
        mse = mean_squared_error(Y_test, predictions)
        r2 = r2_score(Y_test, predictions)
        results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}

        # Log evaluation metrics to W&B
        wandb.log({
            f"{name}_MAE": mae,
            f"{name}_MSE": mse,
            f"{name}_R2": r2
        })

    results_df = pd.DataFrame(results).T
    print(results_df)

    # Optionally, log the results DataFrame as a table
    wandb.log({"evaluation_results": wandb.Table(dataframe=results_df)})

    wandb.finish()

    return results_df