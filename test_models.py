import copy
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet


def make_plot(model, X_train, y_train, X_test, y_test, ax):
    # some models need to have their predicted values reshaped, which seems like
    # it might be a bug
    predicted_train = model.predict(X_train).reshape(X_train.shape)
    residuals_train = predicted_train - y_train

    predicted_test = model.predict(X_test).reshape(X_test.shape)
    residuals_test = predicted_test - y_test

    ax.scatter(predicted_train, residuals_train, c="blue", label="Training Data")
    ax.scatter(predicted_test, residuals_test, c="orange", label="Testing Data")
    ax.legend()
    ax.hlines(y=0,
              xmin=min(predicted_train.min(), predicted_test.min()),
              xmax=max(predicted_train.max(), predicted_test.max()))
    return ax


def go(make_plots=False):
    X, y = make_regression(n_samples=20, n_features=1, random_state=0, noise=6, bias=100.0)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    models = [LinearRegression(), Lasso(alpha=.01), Ridge(alpha=.01), ElasticNet(alpha=.01)]
    scales = [StandardScaler(), RobustScaler()]

    if make_plots:
        fig = plt.figure(figsize=(24, 12))
        subplot_count = 1

    for model in models:
        for scale in scales:
            # copy the scalers so that two separate ones can be made for the
            # independent and dependent variables
            X_scale = copy.deepcopy(scale).fit(X_train)
            y_scale = copy.deepcopy(scale).fit(y_train)

            # scale the training data
            X_train_scaled = X_scale.transform(X_train)
            y_train_scaled = y_scale.transform(y_train)

            # train the model
            model.fit(X_train_scaled, y_train_scaled)

            # scale the testing data
            X_test_scale = X_scale.transform(X_test)
            y_test_scale = y_scale.transform(y_test)

            # score the model with the testing data
            score = model.score(X_test_scale, y_test_scale)
            print(model, scale)
            print(f"\t{score}")
            if make_plots:
                axes = fig.add_subplot(4, 2, subplot_count)
                subplot_count += 1
                print(f"{subplot_count}")
                axes = make_plot(model, X_train_scaled, y_train_scaled, X_test_scale, y_test_scale, axes)
                axes.set_title(f"{model} with {scale},  R2 = {score}")

    if make_plots:
        plt.show()


if __name__ == '__main__':
    go(True)
