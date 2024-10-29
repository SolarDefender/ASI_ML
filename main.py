from analysis import perform_analysis
from training import train_models
from evaluation import evaluate_models

def main():
    #print("Starting analysis...")
    #perform_analysis()
    
    print("\nTraining models...")
    models, X_test, Y_test = train_models()
    
    print("\nEvaluating models...")
    evaluate_models(models, X_test, Y_test)

if __name__ == '__main__':
    main()
