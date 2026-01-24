"""
Hyperparameter Tuning with MLflow Template

This example demonstrates various hyperparameter tuning approaches
with comprehensive MLflow tracking.
"""

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import optuna
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("hyperparameter-tuning")

# Enable autologging
mlflow.sklearn.autolog()


def prepare_data():
    """Generate sample dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


def grid_search_example():
    """Grid search with MLflow tracking"""
    logger.info("Running Grid Search example...")

    X_train, X_test, y_train, y_test = prepare_data()

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10]
    }

    with mlflow.start_run(run_name="grid_search"):
        # Log search configuration
        mlflow.log_param("search_type", "grid_search")
        mlflow.log_param("param_grid", str(param_grid))

        # Create and run grid search
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Evaluate on test set
        y_pred = grid_search.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)

        # Log the best model
        mlflow.sklearn.log_model(
            grid_search.best_estimator_,
            "best_model",
            registered_model_name="GridSearchBestModel"
        )

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")

        return grid_search.best_params_


def random_search_example():
    """Random search with MLflow tracking"""
    logger.info("Running Random Search example...")

    X_train, X_test, y_train, y_test = prepare_data()

    # Define parameter distributions
    param_distributions = {
        'n_estimators': [10, 50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    }

    with mlflow.start_run(run_name="random_search"):
        mlflow.log_param("search_type", "random_search")
        mlflow.log_param("n_iter", 20)

        # Create and run random search
        random_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions,
            n_iter=20,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        # Log results
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_cv_score", random_search.best_score_)

        # Test evaluation
        y_pred = random_search.predict(X_test)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred))

        logger.info(f"Best parameters: {random_search.best_params_}")

        return random_search.best_params_


def nested_runs_manual_search():
    """Manual parameter search with nested MLflow runs"""
    logger.info("Running manual search with nested runs...")

    X_train, X_test, y_train, y_test = prepare_data()

    param_combinations = [
        {'n_estimators': 50, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 7},
        {'n_estimators': 200, 'max_depth': 10},
        {'n_estimators': 150, 'max_depth': 8}
    ]

    best_score = 0
    best_params = None

    # Parent run for the entire search
    with mlflow.start_run(run_name="manual_parameter_search"):
        mlflow.log_param("search_type", "manual_nested")
        mlflow.log_param("num_combinations", len(param_combinations))

        # Child runs for each parameter combination
        for idx, params in enumerate(param_combinations):
            with mlflow.start_run(run_name=f"combination_{idx}", nested=True):
                # Log parameters
                mlflow.log_params(params)

                # Train model
                model = RandomForestClassifier(random_state=42, **params)
                model.fit(X_train, y_train)

                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                mlflow.log_metric("cv_mean", cv_mean)
                mlflow.log_metric("cv_std", cv_std)

                # Test set evaluation
                y_pred = model.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)
                mlflow.log_metric("test_accuracy", test_acc)

                logger.info(f"Params: {params}, CV: {cv_mean:.4f}, Test: {test_acc:.4f}")

                # Track best
                if cv_mean > best_score:
                    best_score = cv_mean
                    best_params = params

        # Log best to parent run
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_score", best_score)

        logger.info(f"Best parameters: {best_params}")
        return best_params


def optuna_integration_example():
    """Optuna hyperparameter optimization with MLflow"""
    logger.info("Running Optuna optimization...")

    X_train, X_test, y_train, y_test = prepare_data()

    def objective(trial):
        """Optuna objective function"""
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }

        # Train model
        model = RandomForestClassifier(random_state=42, **params)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()

        # Log trial to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_accuracy", cv_mean)
            mlflow.log_metric("cv_std", cv_scores.std())
            mlflow.set_tag("trial_number", trial.number)

        return cv_mean

    # Parent run for Optuna study
    with mlflow.start_run(run_name="optuna_optimization"):
        mlflow.log_param("search_type", "optuna")
        mlflow.log_param("n_trials", 30)

        # Create and run study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        # Log best results
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_score", study.best_value)

        # Train final model with best params
        best_model = RandomForestClassifier(random_state=42, **study.best_params)
        best_model.fit(X_train, y_train)

        # Evaluate
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", test_acc)

        # Register model
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            registered_model_name="OptunaOptimizedModel"
        )

        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")
        logger.info(f"Test accuracy: {test_acc:.4f}")

        return study.best_params


def compare_search_methods():
    """Compare different hyperparameter search methods"""
    logger.info("Comparing search methods...")

    with mlflow.start_run(run_name="search_method_comparison"):
        methods = {
            'grid_search': grid_search_example,
            'random_search': random_search_example,
            'manual_nested': nested_runs_manual_search,
            'optuna': optuna_integration_example
        }

        results = {}
        for method_name, method_func in methods.items():
            logger.info(f"\n{'='*60}\nRunning {method_name}\n{'='*60}")
            best_params = method_func()
            results[method_name] = best_params

        # Log comparison
        mlflow.log_dict(results, "comparison_results.json")

        logger.info("\n" + "="*60)
        logger.info("Comparison complete! Check MLflow UI for detailed results")
        logger.info("="*60)


if __name__ == "__main__":
    print("MLflow Hyperparameter Tuning Examples")
    print("="*60)

    # Run individual examples or comparison
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "grid":
            grid_search_example()
        elif example == "random":
            random_search_example()
        elif example == "manual":
            nested_runs_manual_search()
        elif example == "optuna":
            optuna_integration_example()
        elif example == "compare":
            compare_search_methods()
        else:
            print(f"Unknown example: {example}")
            print("Available: grid, random, manual, optuna, compare")
    else:
        # Run all examples
        compare_search_methods()

    print("\nView results at: http://localhost:5000")
