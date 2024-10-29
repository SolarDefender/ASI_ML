import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and inspect data
def perform_analysis(file_path='data/powerconsumption.csv'):
    data = pd.read_csv(file_path)
    print(data.head())
    print(data.info())
    print(data.describe())

    # Check for missing values
    print(data.isnull().sum())
    sns.heatmap(data.isnull(), cbar=False)
    plt.show()

    # Plot histograms
    data.hist(bins=50, figsize=(10, 8))
    plt.show()

    numeric_data = data.select_dtypes(include=['number'])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()  

if __name__ == '__main__':
    perform_analysis()
