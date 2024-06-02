import argparse

# Polyaxon
from polyaxon import tracking

from model import train_and_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=33,
    )
    args = parser.parse_args()

    # Polyaxon
    tracking.init()

    # Train and eval the model with given parameters.
    # Polyaxon
    model_path = "model.joblib"
    metrics = train_and_eval(
        model_path=model_path,
        num_epochs=args.num_epochs,
        metric=args.metric,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Logging metrics to Polyaxon
    print("Testing metrics: {}", metrics)

    # Polyaxon
    tracking.log_metrics(**metrics)

    # Logging the model
    tracking.log_model(
        model_path, name="dnn1_model", framework="tensorflow", versioned=False
    )