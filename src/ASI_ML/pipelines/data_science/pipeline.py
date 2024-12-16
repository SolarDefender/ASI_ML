from kedro.pipeline import Pipeline, node
from .export_parameters import export_parameters_to_gcs
from .perform_analysis import perform_analysis
from .split_data import split_data
from .train_models import train_models
from .evaluate_models import evaluate_models
from .api_run import api_run
def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=export_parameters_to_gcs,
            inputs="parameters",
            outputs=None,
            name="export_parameters_node"
        ),node(
            func=perform_analysis,
            inputs="power_consumption_raw",
            outputs=None,
            name="perform_analysis_node"
        ),
        node(
            func=split_data,
            inputs=["power_consumption_raw", "parameters"],
            outputs=None,
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
            outputs=["model_metrics","best_model"],
            name="evaluate_models_node"
        ),
        node(
            func=api_run,
            inputs=["best_model"],
            outputs=None,
            name="api_node"
        )
    ])