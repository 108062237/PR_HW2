# classifiers/logistic_regression.py

import numpy as np

class LogisticRegressionClassifier:
    def __init__(self, lr=0.1, max_iter=100000, tol=1e-4, verbose=False):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.W = None  # shape: (n_features, n_classes)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # avoid overflow
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.num_classes = np.max(y) + 1
        y_onehot = self._one_hot(y, self.num_classes)

        # 初始化參數 (包含 bias)
        X_bias = np.hstack([X, np.ones((n_samples, 1))])  # 增加 bias 維度
        self.W = np.zeros((n_features + 1, self.num_classes))

        for i in range(self.max_iter):
            logits = X_bias @ self.W
            probs = self._softmax(logits)
            grad = X_bias.T @ (probs - y_onehot) / n_samples

            W_old = self.W.copy()
            self.W -= self.lr * grad

            if self.verbose and i % 100 == 0:
                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))
                print(f"[Iter {i}] Loss: {loss:.4f}")

            if np.linalg.norm(self.W - W_old) < self.tol:
                break

    def predict(self, X):
        n_samples = X.shape[0]
        X_bias = np.hstack([X, np.ones((n_samples, 1))])
        logits = X_bias @ self.W
        probs = self._softmax(logits)
        y_pred = np.argmax(probs, axis=1)
        return y_pred, probs  # return class & discriminant scores (for ROC)
