
from analysis import perform_eda
from training import train_models
from evaluation import evaluate_models

def main():
    print("Starting EDA...")
    perform_eda()
    
    print("\nTraining models...")
    train_models()
    
    print("\nEvaluating models...")
    evaluate_models()

if __name__ == '__main__':
    main()
