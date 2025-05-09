import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.class_stats = {}
        self.class_priors = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.class_stats[c] = {
                "mean": X_c.mean(axis=0),
                "var": X_c.var(axis=0) + 1e-6  
            }
            self.class_priors[c] = len(X_c) / len(X)

    def _gaussian_log_likelihood(self, x, mean, var):
        return -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var, axis=1)

    def predict_proba(self, X):
        log_probs = []
        for c in self.classes:
            mean = self.class_stats[c]["mean"]
            var = self.class_stats[c]["var"]
            log_likelihood = self._gaussian_log_likelihood(X, mean, var)
            log_prior = np.log(self.class_priors[c])
            log_post = log_likelihood + log_prior
            log_probs.append(log_post)

        # shape: (num_samples, num_classes)
        return np.vstack(log_probs).T

    def predict(self, X):
        log_posteriors = self.predict_proba(X)
        preds = np.argmax(log_posteriors, axis=1)

        probs = np.exp(log_posteriors)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        return self.classes[preds], probs
