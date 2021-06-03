from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utility1 import logger_start


def create_pipeline():
    pipeline_Logistic_Regression = Pipeline([('scalar1', StandardScaler()),
                            ('pca1', PCA(n_components=2)),
                            ('lr_classifier', LogisticRegression(random_state=0))])
    pipeline_Decision_Tree = Pipeline([('scalar2', StandardScaler()),
                            ('pca2', PCA(n_components=2)),
                            ('dt_classifier', DecisionTreeClassifier())])
    pipeline_randomforest = Pipeline([('scalar3', StandardScaler()),
                                      ('pca3', PCA(n_components=2)),
                                      ('rf_classifier', RandomForestClassifier())])

    pipelines = [pipeline_Logistic_Regression, pipeline_Decision_Tree, pipeline_randomforest]

    pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest'}

    return pipelines, pipe_dict


def find_classifier_best_Accuracy(X_test, pipe_dict, pipelines, y_test):
    best_accuracy = 0.0
    best_classifier = 0
    for i, model in enumerate(pipelines):
        if model.score(X_test, y_test) > best_accuracy:
            best_accuracy = model.score(X_test, y_test)
            best_pipeline = model
            best_classifier = i

    print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))


def main():
    class_name, method_name = "pipeline.py", main.__name__
    logger_start(class_name, method_name)
    global X_train, X_test, y_train, y_test
    data = load_iris()
    iris_df = pd.DataFrame(data.data, columns=data.feature_names)
    # taking target from predictors
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(iris_df, y, test_size=0.3, random_state=0)

    pipelines, pipe_dict= create_pipeline()

    # Fitting the pipelines
    for pipe in pipelines:
        pipe.fit(X_train, y_train)

    for i, model in enumerate(pipelines):
        print("{} Test Accuracy: {}".format(pipe_dict[i], model.score(X_test, y_test)))

    find_classifier_best_Accuracy(X_test, pipe_dict, pipelines, y_test)


if __name__ == "__main__":
    main()