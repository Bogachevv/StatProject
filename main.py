import numpy as np
import stat_regression as sr
import sklearn.datasets as dt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

if __name__ == '__main__':  # only for testing
    test = dt.fetch_california_housing()
    y, X_prior = test.pop('target', None), test
    X = list()
    for key, item in X_prior.items():
        X.append(item)
    X = np.array(X[0])
    scaler = StandardScaler()
    scaler.fit(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    print(scaler.mean_)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    new_class = sr.RegressionStat(y_test, y_pred)
    new_class.orig_dist()
    new_class.pred_dist()
