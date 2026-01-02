import os
import mlflow
from mlflow import MlflowClient


# def promote_best_model(experiment_name, metric_name):
#     client = MlflowClient()

#     experiment = client.get_experiment_by_name(experiment_name)
#     runs = client.search_runs(experiment.experiment_id)

#     best_run = max(
#         runs,
#         key=lambda r: r.data.metrics.get(metric_name, -1)
#     )

#     model_uri = f"{best_run.info.artifact_uri}/model"
#     # model_uri = f"runs:/{best_run.info.run_id}/model"
#     print("MODEL URI:", model_uri)
#     model_name = experiment_name
#     best_run_metric = best_run.data.metrics.get(metric_name)

#     # client.get_logged_model
#     print(f"Best run found: {best_run.info.run_id} with {metric_name}: {best_run_metric}")

#     try:
#         # Get the version currently tagged as Champion
#         champion_version = client.get_model_version_by_alias(model_name, "Champion")
#         print("CHAMPION:", champion_version)
#         # champion_run = client.get_run(champion_version.run_id)
#         # Check if run_id exists; if not, we might need to fetch by version metadata
#         client.get
#         if not champion_version.run_id:
#             # Fallback: If run_id is empty, get the version details directly
#             champion_version = client.get_model_version(model_name, champion_version.version)
#         champion_run = client.get_run(champion_version.run_id)
#         current_best_metric = float(champion_run.data.metrics.get(metric_name, 0))

#         print(f"Current Champion {metric_name}: {current_best_metric}, New Model {metric_name}: {best_run_metric}")

#         if best_run_metric > current_best_metric:
#             print("New model is better! Promoting...")
#             result = mlflow.register_model(model_uri, model_name)
#             client.set_registered_model_alias(model_name, "Champion", result.version)
#             print(f"Model version {result.version} promoted to 'Champion' alias.")
#         else:
#             print("New model is not an improvement. Skipping registry.")
#     except mlflow.exceptions.RestException:
#         print("No Champion found. Registering first version.")
#         result = mlflow.register_model(model_uri, model_name)
#         client.set_registered_model_alias(model_name, "Champion", result.version)
#         print(f"Model version {result.version} promoted to 'Champion' alias.")


def promote_model(current_run_best_model_info, current_run_best_metrics, artifact_uri):
    model_registered_name = "demo_model"
    client = MlflowClient()
    try:
        reg_model = client.get_model_version_by_alias(alias="Champion", name=model_registered_name)
        registeres_metrics = reg_model.tags
        registeres_metrics = {k:float(v) for k,v in registeres_metrics.items()}

        if current_run_best_metrics['f1_weighted'] > registeres_metrics["f1_weighted"]:
            print("Current experiments yielded a better model. Registering new model")
            print("Current Registered Best:", registeres_metrics)
            print("Current Experiment Best:", current_run_best_metrics)
            model_version = client.create_model_version(
                model_registered_name, 
                artifact_uri, #current_run_best_model_info.model_uri, #uri, 
                tags=current_run_best_metrics
            )
            print(f"New model version {model_version.version} with tags {model_version.tags} created successfully.")
            client.set_registered_model_alias(alias="Champion", name=model_registered_name, version=model_version.version)
        else:
            print("Existing registered model is better")
            print("Current Registered Best:", registeres_metrics)
            print("Current Experiment Best:", current_run_best_metrics)
    except mlflow.exceptions.MlflowException:
        print(f"No models registered with name {model_registered_name}. Registering new model")
        
        client.create_registered_model(model_registered_name, tags=current_run_best_metrics)
        reg_model = client.get_registered_model(model_registered_name)
        model_version = client.create_model_version(
            model_registered_name, 
            artifact_uri, #current_run_best_model_info.model_uri, 
            tags=current_run_best_metrics
        )
        client.set_registered_model_alias(alias="Champion", name=model_registered_name, version=model_version.version)

        # client.copy_model_version(
        #     current_run_best_model_info.model_uri,
        #     model_registered_name
        # )
        
        print(f"New model version {model_version.version} with tags {model_version.tags} created successfully.")