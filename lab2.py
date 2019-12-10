import numpy as np
import pandas as pd
import random
import time
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class RandomForest():
    def __init__(self, max_depth=5, n_trees=3, sample_size=0.1):
        self.trees = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            X_sample, y_sample = self.subsample(X, y, self.sample_size)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros(len(X))

        for tree in self.trees:
            pred = tree.predict(X)
            predictions += pred
        predictions /= len(self.trees)

        return predictions

    def subsample(self, X, y, ratio):
        sample = []
        sample_y = []
        n_sample = round(len(X) * ratio)
        while len(sample) < n_sample:
            index = random.randrange(len(X))
            sample.append(X[index])
            sample_y.append(y[index])
        return np.array(sample), np.array(sample_y)


class DecisionTree():
    def __init__(self, max_depth=5):
        self.feature = None
        self.label = None
        self.n_samples = None
        self.gain = None
        self.root = None
        self.left = None
        self.right = None
        self.threshold = None
        self.depth = 0
        self.max_depth = max_depth

    def fit(self, features, target):
        self.root = DecisionTree()
        self.root.build(features, target)
        self.root.prune(self.max_depth, self.root.n_samples)

    def predict(self, features):
        return np.array([self.root.predict_feature(feature) for feature in features])

    def predict_feature(self, feature):
        if self.feature != None:
            if feature[self.feature] <= self.threshold:
                return self.left.predict_feature(feature)
            else:
                return self.right.predict_feature(feature)
        else:
            return self.label

    def build(self, features, target):
        self.n_samples = features.shape[0]

        if len(np.unique(target)) == 1:
            self.label = target[0]
            return

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        self.label = np.mean(target)

        impurity_node = self.mse(target)

        for col in range(features.shape[1]):
            feature_level = np.unique(features[:, col])
            thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0

            for threshold in thresholds:
                target_l = target[features[:, col] <= threshold]
                impurity_l = self.mse(target_l)
                n_l = float(target_l.shape[0]) / self.n_samples

                target_r = target[features[:, col] > threshold]
                impurity_r = self.mse(target_r)
                n_r = float(target_r.shape[0]) / self.n_samples

                impurity_gain = impurity_node - (n_l * impurity_l + n_r * impurity_r)
                if impurity_gain > best_gain:
                    best_gain = impurity_gain
                    best_feature = col
                    best_threshold = threshold

        self.feature = best_feature
        self.gain = best_gain
        self.threshold = best_threshold
        self.split_node(features, target)

    def split_node(self, features, target):
        features_l = features[features[:, self.feature] <= self.threshold]
        target_l = target[features[:, self.feature] <= self.threshold]
        self.left = DecisionTree()
        self.left.depth = self.depth + 1
        self.left.build(features_l, target_l)

        features_r = features[features[:, self.feature] > self.threshold]
        target_r = target[features[:, self.feature] > self.threshold]
        self.right = DecisionTree()
        self.right.depth = self.depth + 1
        self.right.build(features_r, target_r)

    def mse(self, target):
        return np.mean((target - np.mean(target)) ** 2)

    def prune(self, max_depth, n_samples):
        if self.feature is None:
            return

        self.left.prune(max_depth, n_samples)
        self.right.prune(max_depth, n_samples)

        if self.depth >= max_depth:
            self.left = None
            self.right = None
            self.feature = None



class PolynomialRegression:
    def __init__(self, order):
        self.poly = PolynomialFeatures(order)
    
    def fit(self, X, Y):
        transformed_X = self.poly.fit_transform(X)
        self.coef = np.linalg.lstsq(transformed_X, Y, rcond=None)[0]
    
    def predict(self, X):
        return np.dot(self.poly.fit_transform(X), self.coef)


def split_data(data, labels, test_percent):
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for row, label in zip(data, labels):
        if random.random() < test_percent:
            test_data.append(row)
            test_label.append(label)
        else:
            train_data.append(row)
            train_label.append(label)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    return (train_data, train_label), (test_data, test_label)

if __name__ == "__main__":
    data = pd.read_csv("Tesla.csv")
    X = np.array(data.drop(columns='Date'))
    y = np.array(data.pop('Close'))

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.3)

    X = np.array(X)
    X_test = np.array(X_test)
    y = np.array(y)
    y_test = np.array(y_test)

    for n_trees in [2, 5, 10, 20, 50]:
        begin = time.time()

        reg = RandomForest(max_depth=5, n_trees=n_trees)
        reg.fit(X, y)
        y_pred = reg.predict(X_test)

        end = time.time()

        time1 = end - begin

        begin = time.time()

        sk_reg = RandomForestRegressor(max_depth=5, n_estimators=n_trees)
        sk_reg.fit(X, y)
        sk_y_pred = sk_reg.predict(X_test)

        end = time.time()

        time2 = end - begin

        print("{}{:<25s}".format(n_trees, " trees"), "mine ", "sklearn")
        print("{:<26s}{} {}".format("time: ", round(time1, 4), round(time2, 4)))
        print("{:<25s}".format("Explained_variance_score:"), round(metrics.explained_variance_score(y_test, y_pred), 4),
              round(metrics.explained_variance_score(y_test, sk_y_pred), 4))
        print("{:<25s}".format("Mean Absolute Error:"), round(metrics.mean_absolute_error(y_test, y_pred), 4),
              round(metrics.mean_absolute_error(y_test, sk_y_pred), 4))
        print("{:<25s}".format("Mean Squared Error:"), round(metrics.mean_squared_error(y_test, y_pred), 4),
              round(metrics.mean_squared_error(y_test, sk_y_pred), 4))
        print()


    df = pd.read_csv('Tesla-redacted.csv')

    prices = np.array(df[['Date', 'Open', 'High', 'Low', 'Volume']].to_numpy())
    features = df['Close'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(prices, features, test_size=0.3)
    
    for i in range(1, 4):
        start_time = time.perf_counter_ns()
        prd = PolynomialRegression(i)
        prd.fit(X_train, y_train)
        y_pred = prd.predict(X_test)
        end_time = time.perf_counter_ns()
        
        print('Polynomial Regression of {} order'.format(i))
        print('Time: ', (end_time - start_time) / 10 ** 9)
        print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print()    
