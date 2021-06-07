import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def separate_num_cat_data(train_df):
    df_train_num = train_df.select_dtypes(include=['int64', 'float64'])
    df_train_txt = train_df.select_dtypes(include=['object'])
    return df_train_num, df_train_txt


def filter_df(test_df):
    test_df = test_df.mask(test_df.applymap(str).eq('[]'))
    na_percent = (test_df.isnull().sum() / len(test_df)) * 100
    cols_drop = na_percent[na_percent > 70].keys().tolist()

    return test_df, cols_drop


def separate_num_cat_columns(X_train_full):
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() and
                        X_train_full[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    return categorical_cols, numerical_cols


def numeric_data_completeness(train_df):
    null_values = train_df.isna().sum()
    null_values = null_values[null_values != 0]
    print("NULL Values: ", null_values)
    columns = null_values.index.get_level_values(0).to_list()
    for column in columns:
        train_df[column].fillna(train_df[column].mean(), inplace=True)
        # train_df[column].fillna(train_df[column].mode()[0],inplace=True)

    return train_df


def create_pipeline(numerical_cols, categorical_cols):
    # Preprocessing for numerical data
    # numerical_transformer = SimpleImputer(strategy='constant')
    numerical_transformer = Pipeline(steps=[('numeric_imputer', SimpleImputer(strategy='constant'))])
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('categoric_imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', LinearRegression())
                                  ])

    return my_pipeline


def main():
    test_df = pd.read_csv('dummy_data.csv')
    print(" Data Columns Name: %s" % test_df.columns.values.tolist())
    print(" Data read completed")

    # Separate target from predictors
    y = test_df.Price
    X = test_df.drop(['Price'], axis=1)

    test_df, dropped_cols = filter_df(test_df)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                    random_state=0)

    print("Separate numeric and categorical columns")
    categorical_cols, numerical_cols = separate_num_cat_columns(X)

    pipeline = create_pipeline(numerical_cols, categorical_cols)

    pickle_file = pipeline.fit(xtrain, ytrain)

    joblib.dump(pickle_file, "LR_pickle.pkl")

    pipeline_pickle = joblib.load("LR_pickle.pkl")

    # Preprocessing of validation data, get predictions
    preds = pipeline_pickle.predict(xtest)

    # Root Mean Squared Error on train and test date
    # score = mean_absolute_error(ytest, preds)
    score = pipeline_pickle.score(xtest, ytest)
    print("score: ", score)

    rmse_score = mean_squared_error(ytest, preds, squared=False)
    print("Root Mean Squared Error: ", rmse_score)


if __name__ == "__main__":
    main()
