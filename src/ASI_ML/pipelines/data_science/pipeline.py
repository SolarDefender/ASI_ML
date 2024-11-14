from kedro.pipeline import Pipeline, node
from .nodes import perform_analysis, split_data, train_models, evaluate_models

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=perform_analysis,
            inputs="power_consumption_raw",
            outputs=None,
            name="perform_analysis_node"
        ),
        node(
            func=split_data,
            inputs=["power_consumption_raw", "parameters"],
            outputs=["X_train", "X_dev", "X_test", "Y_train", "Y_dev", "Y_test"],
            name="split_data_node"
        ),
        node(
            func=train_models,
            inputs=["X_train", "Y_train", "X_dev", "Y_dev", "parameters"],
            outputs="trained_models",
            name="train_models_node"
        ),
        node(
            func=evaluate_models,
            inputs=["trained_models", "X_test", "Y_test", "parameters"],
            outputs="model_metrics",
            name="evaluate_models_node"
        )
    ])
