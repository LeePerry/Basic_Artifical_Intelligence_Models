#!/usr/bin/env python3

# standard library
import os
import sys
import argparse
import csv

# matplotlib
import matplotlib.pyplot as plt

# numpy
from numpy import mean
from numpy import array as np_array

# sklearn
from sklearn.dummy import DummyRegressor as dummy_regressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold as k_fold
from sklearn.neighbors import KNeighborsRegressor as kn_regressor
from sklearn.neural_network import MLPRegressor as mlp_regressor
from sklearn.linear_model import LinearRegression as linear_regressor


# parse all command-line arguments, validate and return them as an object
def command_line_args():

    # parse command-line (also prints help if required)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="EMATM0044 Introduction to AI: Coursework Question 1\n"+
                                                 "Student Number: 2315517")
    parser.add_argument("-d",
                        "--data_path",
                        type=str,
                        required=True,
                        help="The full path to a CSV dataset file")
    parser.add_argument("-f",
                        "--output_path",
                        type=str,
                        default=None,
                        help="[OPTIONAL] The path in which to store the output graphs and raw data. Defaults to <data_path>/../output")
    args = parser.parse_args()

    # if the output directory has not been provided, then default it relative to the dataset
    if not args.output_path:
        args.output_path = os.path.join(
            os.path.abspath(os.path.join(args.data_path, os.pardir)), "output")

    # validate that the dataset path actually exists
    if not os.path.exists(args.data_path):
        parser.print_help()
        print("\n\nPlease provide a valid data path!\n\n")
        sys.exit(1)

    return args


# load the data from the path specified via args and extract feature names
# also splits data into training and testing
def load_data(args):

    # open the dataset file
    with open(args.data_path, "r") as f:
        data = []
        target = []
        reader = csv.reader(f)

        # read the column headers (i.e. feature/target names)
        for row in reader:
            feature_labels = row
            break

        # read data and split into features and targets (last assumed to be target)
        for row in reader:
            data.append(np_array([float(n) for n in row[:-1]]))
            target.append(float(row[-1]))
        data = np_array(data)
        target = np_array(target)
        return [feature_labels] + train_test_split(data, target, test_size=0.2)


# perform the 1d grid search over hyperparameter options for the baseline
# store the performance results both as raw data and graph
def compare_hyperparameters_for_baseline(args, x_train, y_train):
    strategies = ["mean", "median"] + [f"quantile={q}" for q in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    s_mse = []

    # iterate over hyperparameter 1 options: strategies
    for strategy in strategies:
        mse = []

        # split the training data into 20 folds
        kf = k_fold(n_splits=20, shuffle=True, random_state=42)
        for train_index, val_index in kf.split(x_train):
            x_train_split, x_val_split = x_train[train_index], x_train[val_index]
            y_train_split, y_val_split = y_train[train_index], y_train[val_index]

            # train model and make prediction
            y_pred = predict_using_baseline(x_train_split,
                                                   x_val_split,
                                                   y_train_split,
                                                   strategy=strategy)

            # calculate MSE for this fold
            mse.append(mean_squared_error(y_val_split, y_pred))

        # average MSE over all 20 folds
        s_mse.append(mean(mse))

    # print raw data to stdout.txt in output directory
    print("Baseline Hyperparameters - Mean Squared Error:")
    print(','.join(strategies))
    print(','.join(["{:.3f}".format(mse) for mse in s_mse]))

    # plot the hyperparameter vs MSE as a graph and save to output directory
    _, ax = plt.subplots()
    plt.xlabel("Mean MSE Over 20 Folds")
    bars = plt.barh(strategies, s_mse)
    ax.bar_label(bars)
    plt.tight_layout()
    plt.xticks(range(0, 2600, 250))
    plt.savefig(os.path.join(args.output_path, "baseline_hp_mse.png"))


# perform the 2d grid search over hyperparameter options for the knn model
# store the performance results both as raw data and graph
def compare_hyperparameters_for_knn_regressor(args, x_train, y_train):
    values_of_k = [i for i in range(1, 21)]
    tabular_mse = [["Weights"] + [f"K={k}" for k in values_of_k]]
    plot_mse = []

    # iterate over hyperparameter 1 options: weights
    for weights, colour in [("uniform",  "blue"),
                            ("distance", "orange")]:
        k_mse = []

        # iterate over hyperparameter 2 options: k
        for k in values_of_k:
            mse = []

            # split the training data into 20 folds
            kf = k_fold(n_splits=20, shuffle=True, random_state=42)
            for train_index, val_index in kf.split(x_train):
                x_train_split, x_val_split = x_train[train_index], x_train[val_index]
                y_train_split, y_val_split = y_train[train_index], y_train[val_index]

                # train model and make prediction
                y_pred = predict_using_knn_regressor(x_train_split,
                                                    x_val_split,
                                                    y_train_split,
                                                    neighbours=k,
                                                    weights=weights)

                # calculate MSE for this fold
                mse.append(mean_squared_error(y_val_split, y_pred))

            # average MSE over all 20 folds
            k_mse.append(mean(mse))
        plot_mse.append((k_mse, colour, weights))
        tabular_mse.append([weights] + k_mse)

    # print raw data to stdout.txt in output directory
    print("KNN Hyperparameters - Mean Squared Error:")
    print(','.join(tabular_mse[0]))
    for row in tabular_mse[1:]:
        print(','.join([row[0]] + ["{:.3f}".format(mse) for mse in row[1:]]))

    # plot the hyperparameter vs MSE as a graph and save to output directory
    _, ax = plt.subplots()
    plt.xticks(values_of_k)
    plt.xlabel("K")
    plt.ylabel("Mean MSE Over 20 Folds")
    for k_mse, colour, activation_function in plot_mse:
        ax.plot(values_of_k, k_mse, color=colour, label=activation_function)
    plt.legend(title="Weights")
    plt.savefig(os.path.join(args.output_path, "knn_hp_mse.png"))


# perform the 2d grid search over hyperparameter options for the mlp model
# store the performance results both as raw data and graph
def compare_hyperparameters_for_mlp_regressor(args, x_train, y_train):
    number_of_nodes = [i for i in range(1, 11)]
    tabular_mse = [["Activation Function"] + [f"{n} Nodes" for n in number_of_nodes]]
    plot_mse = []

    # iterate over hyperparameter 1 options: activtation function
    for activation_function, colour in [("tanh",     "yellow"), # similar mse as logistic
                                        ("logistic", "green"),
                                        ("identity", "blue"),
                                        ("relu",     "red")]:
        n_nodes_mse = []

        # iterate over hyperparameter 2 options: number of nodes in hidden layer
        for i in number_of_nodes:
            mse = []

            # split the training data into 20 folds
            kf = k_fold(n_splits=20, shuffle=True, random_state=42)
            for train_index, val_index in kf.split(x_train):
                x_train_split, x_val_split = x_train[train_index], x_train[val_index]
                y_train_split, y_val_split = y_train[train_index], y_train[val_index]

                # train model and make prediction
                y_pred = predict_using_mlp_regressor(x_train_split,
                                                     x_val_split,
                                                     y_train_split,
                                                     hidden_layer_sizes=[i],
                                                     activation_function=activation_function)

                # calculate MSE for this fold
                mse.append(mean_squared_error(y_val_split, y_pred))

            # average MSE over all 20 folds
            n_nodes_mse.append(mean(mse))
        plot_mse.append((n_nodes_mse, colour, activation_function))
        tabular_mse.append([activation_function] + n_nodes_mse)

    # print raw data to stdout.txt in output directory
    print("MLP Hyperparameters - Mean Squared Error:")
    print(','.join(tabular_mse[0]))
    for row in tabular_mse[1:]:
        print(','.join([row[0]] + ["{:.3f}".format(mse) for mse in row[1:]]))

    # plot the hyperparameter vs MSE as a graph and save to output directory
    _, ax = plt.subplots()
    plt.xticks(number_of_nodes)
    plt.xlabel("Nodes in Hidden Layer")
    plt.ylabel("Mean MSE Over 20 Folds")
    for n_nodes_mse, colour, weights in plot_mse:
        ax.plot(number_of_nodes, n_nodes_mse, color=colour, label=weights)
    plt.legend(title="Activation Function")
    plt.savefig(os.path.join(args.output_path, "mlp_hp_mse.png"))


# train the baseline model and make a prediction
def predict_using_baseline(x_train, x_test, y_train, strategy="mean"):
    dummy = (dummy_regressor(strategy="quantile", quantile=float(strategy.split('=')[1]))
             if strategy.startswith("quantile=") else
             dummy_regressor(strategy=strategy))
    model = dummy.fit(x_train, y_train)
    return model.predict(x_test)


# train the linear model and make a prediction
def predict_using_linear_regression(x_train, x_test, y_train):
    l     = linear_regressor()
    model = l.fit(x_train, y_train)
    return model.predict(x_test)


# train the knn model and make a prediction
def predict_using_knn_regressor(x_train, x_test, y_train, neighbours=6, weights="distance"):
    knn   = kn_regressor(n_neighbors=neighbours, weights=weights)
    model = knn.fit(x_train, y_train)
    return model.predict(x_test)


# train the mlp model and make a prediction
def predict_using_mlp_regressor(x_train,
                                x_test,
                                y_train,
                                hidden_layer_sizes=[1],
                                activation_function="identity",
                                print_coefs=False):
    nn    = mlp_regressor(hidden_layer_sizes=hidden_layer_sizes,
                          random_state=63,
                          max_iter=5000,
                          solver="lbfgs",
                          activation=activation_function)
    model = nn.fit(x_train, y_train)

    # if requested, print the MLP weights to stdout.txt
    if print_coefs:
        print(f"MLP_COEFS:{activation_function},{hidden_layer_sizes},{model.coefs_}")
    return model.predict(x_test)


# for each feature in the dataset, plot the true data and predictions from each of the models
# as a scatter graph, and save it to the output directory
def plot_each_feature_vs_prediction(args, feature_labels, x_test, y_test, dummy_y_pred, knnr_y_pred, mlpr_y_pred):
    size = 2
    marker = "o"
    for feature in range(len(x_test[0])):
        feature_x = [row[feature] for row in x_test]
        _, ax = plt.subplots()
        plt.xlabel(feature_labels[feature])
        plt.ylabel(feature_labels[-1])
        ax.scatter(feature_x, dummy_y_pred, s=size, c="r", marker=marker, label="Baseline")
        ax.scatter(feature_x, knnr_y_pred, s=size, c="y", marker=marker, label="KNN Regressor")
        ax.scatter(feature_x, mlpr_y_pred, s=size, c="b", marker=marker, label="MLP Regressor")
        ax.scatter(feature_x, y_test, s=size, c="g", marker=marker, label="True Data")
        plt.legend(title="Model")
        plt.savefig(os.path.join(args.output_path, f"feature_{feature_labels[feature]}.png"))


# plot each of the models predictions relative to each other and save it the output directory
def plot_comparative_performance(args, y_test, dummy_y_pred, knnr_y_pred, mlpr_y_pred, lr_y_pred):
    print("Displaying Model Performance")
    _, ax = plt.subplots()
    values = [mean_squared_error(y_test, pred) for pred in [dummy_y_pred, lr_y_pred, knnr_y_pred, mlpr_y_pred]]
    plt.xlabel("Mean Squared Error")
    labels = ["Baseline", "Linear\nRegressor", "KNN Regressor", "MLP Regressor"]
    bars = plt.barh(labels, values)
    ax.bar_label(bars)
    plt.tight_layout()
    plt.xticks(range(0, int(max(values) * 1.4), 50))
    plt.savefig(os.path.join(args.output_path, "results_mse.png"))


# the main entry point, invoked when run from the command line
def main():

    # parse command line arguments and prepare the output directory
    args = command_line_args()
    os.makedirs(args.output_path)
    print(f"Output Directory: {args.output_path}")
    sys.stdout = open(os.path.join(args.output_path, "stdout.txt"), 'w')

    # load the dataset
    features, x_train, x_test, y_train, y_test = load_data(args)

    # perform grid searches of hyperparameter options
    compare_hyperparameters_for_baseline(args, x_train, y_train)
    compare_hyperparameters_for_mlp_regressor(args, x_train, y_train)
    compare_hyperparameters_for_knn_regressor(args, x_train, y_train)

    # train all models using the training data and make predictions using the validation data
    dummy_y_pred = predict_using_baseline(x_train, x_test, y_train)
    knnr_y_pred = predict_using_knn_regressor(x_train, x_test, y_train)
    mlpr_y_pred = predict_using_mlp_regressor(x_train, x_test, y_train, print_coefs=True)
    lr_y_pred = predict_using_linear_regression(x_train, x_test, y_train)

    # plot all model predictions and performance comparisons
    plot_each_feature_vs_prediction(args, features, x_test, y_test, dummy_y_pred, knnr_y_pred, mlpr_y_pred)
    plot_comparative_performance(args, y_test, dummy_y_pred, knnr_y_pred, mlpr_y_pred, lr_y_pred)

if __name__ == "__main__":
    main()
