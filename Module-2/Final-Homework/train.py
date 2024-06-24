import os
import pickle
import mlflow
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
mlflow.set_tracking_uri("sqlite:///mlflow.db:5000")
mlflow.set_experiment("RandomForest")


@click.command()
@click.option(
    "--data_path",
    default="/workspaces/codespaces-jupyter/Module-2/Final-Homework/output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.sklearn.autolog()
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_artifact(local_path="/workspaces/codespaces-jupyter/Module-2/Final-Homework/artifacts", artifact_path="models_pickle")

if __name__ == '__main__':
    run_train()