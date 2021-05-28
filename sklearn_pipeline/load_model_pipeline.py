import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


test_df = pd.read_csv('pipeline_dummy_data.csv')
print(" Data Columns Name: %s" % test_df.columns.values.tolist())
print(" Data read completed")

# Separate target from predictors
y = test_df.Price
X = test_df.drop(['Price'], axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                random_state=0)


loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
Ypredict = loaded_model.predict(xtest)
#result = loaded_model.score()
print(Ypredict)