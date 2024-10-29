
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from training import train_models

# Evaluation function
def evaluate_models():
    models, X_test, Y_test = train_models()

    # Evaluate each model
    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        mae = mean_absolute_error(Y_test, predictions)
        mse = mean_squared_error(Y_test, predictions)
        r2 = r2_score(Y_test, predictions)
        results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    print(results_df)

    # Plot results
    metrics = ['MAE', 'MSE', 'R2']
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        plt.bar(results_df.index, results_df[metric])
        plt.title(metric)
    plt.show()

if __name__ == '__main__':
    evaluate_models()
