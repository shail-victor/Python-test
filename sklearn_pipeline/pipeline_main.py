import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import joblib
from utility import logger_start


def seperate_num_cat_data(train_df):
    df_train_num = train_df.select_dtypes(include=['int64', 'float64'])
    df_train_txt = train_df.select_dtypes(include=['object'])
    return df_train_num, df_train_txt


def filter_df(test_df):
    test_df = test_df.mask(test_df.applymap(str).eq('[]'))
    na_percent = (test_df.isnull().sum() / len(test_df)) * 100
    cols_drop = na_percent[na_percent > 70].keys().tolist()
    test_df_cols_droped = test_df[cols_drop]
    test_df_cols_droped = test_df_cols_droped.fillna('')
    test_df.drop(labels=cols_drop, axis=1, inplace=True)
    #train_df = test_df.dropna().reset_index(drop=True)
    col_names = list(test_df.columns.values)
    col_names = col_names + cols_drop

    return test_df, col_names, test_df_cols_droped




def seperate_num_cat_columns(X_train_full):
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    return categorical_cols, numerical_cols


def numeric_data_completeness(train_df):
    null_values=train_df.isna().sum()
    null_values= null_values[null_values!=0]
    print("NULL Values: ", null_values)
    columns = null_values.index.get_level_values(0).to_list()
    for column in columns:
        train_df[column].fillna(train_df[column].mean(),inplace=True)
        # train_df[column].fillna(train_df[column].mode()[0],inplace=True)

    return train_df


def preprocessing_data(numerical_cols,categorical_cols):
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return numerical_transformer, categorical_transformer, preprocessor


def create_fit_pipeline(preprocessor, model, X_train, y_train, X_valid, y_valid):
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                  ])

    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)

    return y_valid, preds



def main():
    class_name, method_name = "pipeline.py", main.__name__
    logger_start(class_name, method_name)
    test_df = pd.read_csv('pipeline_dummy_data.csv')
    print(" Data Columns Name: %s" % test_df.columns.values.tolist())
    print(" Data read completed")

    print("Divide data into training and validation subsets")

    # Separate target from predictors
    y = test_df.Price
    X = test_df.drop(['Price'], axis=1)

    print("Seprate numeric and categorical columns")
    categorical_cols, numerical_cols = seperate_num_cat_columns(X)


    xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                    random_state=0)


    # train_df, col_names, test_df_cols_droped = filter_df(test_df)
    # test_df_cols_droped.to_csv("dropped_columns.csv",encoding="utf-8-sig", index = False)
    #
    # train_df.to_csv("train_data.csv", encoding="utf-8-sig")
    # test_df.to_csv("test_data.csv", encoding="utf-8-sig")



    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = xtrain[my_cols].copy()
    X_valid = xtest[my_cols].copy()

    numerical_transformer, categorical_transformer, preprocessor= preprocessing_data(numerical_cols,categorical_cols)
    model = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=6)
    ytest, preds=create_fit_pipeline(preprocessor, model, X_train, ytrain, X_valid, ytest)

    # Evaluate the model
    score = mean_absolute_error(ytest, preds)
    print("Mean Absolute error score : ", score)

    filename = 'pipeline_pickle_model.pmml'
    joblib.dump(model,filename)
    pipelinem=joblib.load(filename)
    print(pipelinem.score(xtest,ytest))
    #pred=pd.Series(pipelinem.predict(xtest))
    #print("pickle model result: ",pred)

    # pickle.dump(model, open(filename, 'wb'))
    #
    # loaded_model = pickle.load(open("pickle_model.pkl", 'rb'))
    # result = loaded_model.predict(xtest)
    # print("pickle model result: ", result)





    # df_train_num =numeric_data_completeness(df_train_num)
    # df_train_num.to_csv("train_numeric_data.csv", encoding="utf-8-sig", index=False)
    # df_train_txt.to_csv("train_categoric_data.csv", encoding="utf-8-sig", index = False)
    # print("len(df_train_num): %d, len(df_train_txt): %d" % (len(df_train_num), len(df_train_txt)))




if __name__ == "__main__":
    main()
