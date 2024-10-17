from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()

# Replace with the run_id of the run where the model was logged
run_id = "db7cd3cf0cfb446b852e311ea8e870d4"

# get model_path from artifacts of mlflow ui
model_path = "file:///D:/MLOps_CampusX/MLFlow/model-registry-mlflow-demo/mlruns/309391168614291834/db7cd3cf0cfb446b852e311ea8e870d4/artifacts/best_random_forest/model.pkl"

# Construct the model URI
model_uri = f"runs:/{run_id}/{model_path}"

# Register the model in the model registry
model_name = 'diabetes-rf'
result = mlflow.register_model(model_uri, model_name)

import time
time.sleep(3)

# Add a description to the registered model version
client.update_model_version(
    name= model_name,
    version = result.version,
    description = "This is a RandomForest model trained to predict diabetes outcomes based on Pima Indians Diabetes Dataset."    
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="experiment",
    value="diabetes prediction"
)
# add another tag of date
client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="date",
    value="17th 0ct 2024"
)

print(f"Model registered with name: {model_name} and version {result.version}")
print(f"Added tag to model {model_name} version {result.version}")

# Get and print the registered model information
registered_model = client.get_registered_model(model_name)
print("Registered Model Information : ")
print(f"Name : {registered_model.name}")
print(f"Creation Timestamp : {registered_model.creation_timestamp}")
print(f"Last Updated timestamp : {registered_model.last_updated_timestamp}")
print(f"Description : {registered_model.description}")
