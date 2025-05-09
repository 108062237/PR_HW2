import numpy as np
import time
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import load_config, load_dataset
from naive_bayes import NaiveBayesClassifier
from logistic_regression import LogisticRegressionClassifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/breast_cancer.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    return args

def run_pca_classification_experiment(X_train, y_train, X_test, y_test,
                                      model_class_instance,
                                      n_classes=2, model_name=""):
    clf = model_class_instance

    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred, y_score = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    auc_val = None
    if n_classes == 2 and y_score is not None:
        try:
            if y_score.ndim == 2 and y_score.shape[1] >= 2:
                auc_val = roc_auc_score(y_test, y_score[:, 1])
            elif y_score.ndim == 1:
                auc_val = roc_auc_score(y_test, y_score)
        except Exception:
            pass

    return accuracy, auc_val, train_time

def main():
    args = get_args()
    config_path = args.config
    try:
        base_name = os.path.basename(config_path)
        dataset_name = os.path.splitext(base_name)[0]
    except Exception:
        dataset_name = "UnknownDataset"

    print(f"===== Task 2: PCA and Classification on {dataset_name} =====")

    cfg = load_config(config_path)
    X_train_pd, X_test_pd, y_train_pd, y_test_pd = load_dataset(cfg)

    X_train_orig_np = X_train_pd.values
    X_test_orig_np = X_test_pd.values
    y_train_np = y_train_pd.values.ravel()
    y_test_np = y_test_pd.values.ravel()

    n_classes = len(np.unique(y_train_np))
    current_random_seed = cfg.get("random_seed", 42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig_np)
    X_test_scaled = scaler.transform(X_test_orig_np)
    print(f"\nOriginal number of features: {X_train_scaled.shape[1]}")

    classifiers_to_evaluate = {
        "Naive Bayes": NaiveBayesClassifier,
        "Logistic Regression": LogisticRegressionClassifier
    }

    all_experiment_results = []

    for model_name, ModelClass in classifiers_to_evaluate.items():
        print(f"\n===== Using Classifier: {model_name} =====")

        print(f"\n--- {model_name} @ Original Data (No PCA) ---")
        model_instance_orig = ModelClass()
        acc_orig, auc_orig, time_orig = run_pca_classification_experiment(
            X_train_scaled, y_train_np, X_test_scaled, y_test_np,
            model_class_instance=model_instance_orig, n_classes=n_classes, model_name=model_name + " (Original)"
        )
        all_experiment_results.append({
            "Classifier": model_name,
            "PCA Setting": "No PCA",
            "Reduced Dim": X_train_scaled.shape[1],
            "Accuracy": acc_orig,
            "AUC": auc_orig,
            "Training Time(s)": time_orig
        })
        print(f"  Accuracy: {acc_orig:.4f}" + (f", AUC: {auc_orig:.4f}" if auc_orig is not None else ""))
        print(f"  Training Time: {time_orig:.4f}s")

        pca_configurations = [
            {"name": "PCA (retain 95% variance)", "pca_params": {"n_components": 0.95, "random_state": current_random_seed}},
            {"name": "PCA (retain 90% variance)", "pca_params": {"n_components": 0.90, "random_state": current_random_seed}},
        ]
        if X_train_scaled.shape[1] > 2:
            pca_configurations.append({"name": "PCA (k=2 components)", "pca_params": {"n_components": 2, "random_state": current_random_seed}})

        for config_pca in pca_configurations:
            print(f"\n--- {model_name} @ {config_pca['name']} ---")
            pca = PCA(**config_pca['pca_params'])

            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            actual_components = X_train_pca.shape[1]
            print(f"  Reduced dimension after PCA: {actual_components}")
            if isinstance(config_pca['pca_params']['n_components'], float):
                explained_variance_ratio_sum = np.sum(pca.explained_variance_ratio_)
                print(f"  Cumulative explained variance ratio: {explained_variance_ratio_sum:.4f}")

            model_instance_pca = ModelClass()
            acc_pca, auc_pca, time_pca = run_pca_classification_experiment(
                X_train_pca, y_train_np, X_test_pca, y_test_np,
                model_class_instance=model_instance_pca, n_classes=n_classes, model_name=f"{model_name} ({config_pca['name']})"
            )
            all_experiment_results.append({
                "Classifier": model_name,
                "PCA Setting": config_pca['name'],
                "Reduced Dim": actual_components,
                "Accuracy": acc_pca,
                "AUC": auc_pca,
                "Training Time(s)": time_pca
            })
            print(f"  Accuracy: {acc_pca:.4f}" + (f", AUC: {auc_pca:.4f}" if auc_pca is not None else ""))
            print(f"  Training Time: {time_pca:.4f}s")

    print("\n===== Task 2 Summary of Experimental Results =====")
    header = f"{'Classifier':<20} | {'PCA Setting':<28} | {'Reduced Dim':<12} | {'Accuracy':<10} | {'AUC':<10} | {'Training Time(s)':<12}"
    print(header)
    print("-" * (len(header) + 5))
    for res in all_experiment_results:
        auc_str = f"{res['AUC']:.4f}" if res['AUC'] is not None else "N/A"
        acc_str = f"{res['Accuracy']:.4f}"
        time_str = f"{res['Training Time(s)']:.4f}"
        print(f"{res['Classifier']:<20} | {res['PCA Setting']:<28} | {res['Reduced Dim']:<12} | {acc_str:<10} | {auc_str:<10} | {time_str:<12}")

    print("===== Task 2 Script Execution Completed =====")

if __name__ == "__main__":
    main()
