from azureml.core.model import Model
from ml_service.util.env_variables import Env
from azureml.core import Workspace


def main():
    # Register model
    e = Env()
    # Get Azure machine learning workspace
    # ws = Workspace.get(
    #     name=e.workspace_name,
    #     subscription_id=e.subscription_id,
    #     resource_group=e.resource_group
    # )

    ws = Workspace.get(
        name="innovationaiml01",
        subscription_id="1b7772b5-e1ea-49f0-8027-9fd1f6203aa1",
        resource_group="InnovationAIML"
    )

    model = Model.register(workspace=ws,
                           model_path="../../fusion/models/diabetes.pkl",
                           model_name="sklearn-diabetes",
                           tags={"area": "diabetes", "type": "regression"},
                           version=3,
                           description="sklearn Ridge regression model to predict diabetes")


if __name__ == '__main__':
    main()

