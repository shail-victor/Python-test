import numpy as np
import pandas as pd
from faker.providers.person.en import Provider

size = 1000
numeric_range = 10000


def random_names(name_type):
    names = getattr(Provider, name_type)
    return np.random.choice(names, size=size)


def random_genders():
    p = (0.49, 0.49, 0.01, 0.01)
    gender = ("M", "F", "O", "")
    return np.random.choice(gender, size=size, p=p)


def random_integer():
    int_array = np.random.choice(np.arange(numeric_range), size=size)
    return int_array


def random_float():
    float_array = np.round(np.random.uniform(10.1, 2098.5, size=size), 2)
    return float_array


def random_bool():
    p = 0.1
    bool_array = np.random.choice(a=[False, True], size=size, p=[p, 1 - p])
    return bool_array


def random_dates(start, end):
    divide_by = 24 * 60 * 60 * 10 ** 9
    start_u = start.value // divide_by
    end_u = end.value // divide_by
    date_array = pd.to_datetime(np.random.randint(start_u, end_u, size), unit='D', utc=False)
    return date_array


dummy_dataset = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'Name': pd.Series([], dtype='str'),
                              'DOB': pd.Series([], dtype='datetime64[ns]'), 'd': pd.Series([], dtype='int32'),
                              'e': pd.Series([], dtype='int32'), 'f': pd.Series([], dtype='float64'),
                              'g': pd.Series([], dtype='float64'), 'h': pd.Series([], dtype='float64'),
                              'i': pd.Series([], dtype='float64'), 'j': pd.Series([], dtype='float64'),
                              'k': pd.Series([], dtype='float64'), 'l': pd.Series([], dtype='float64'),
                              'm': pd.Series([], dtype='float64'), 'n': pd.Series([], dtype='float64'),
                              'o': pd.Series([], dtype='float64'), 'p': pd.Series([], dtype='float64'),
                              'q': pd.Series([], dtype='float64'), 'r': pd.Series([], dtype='float64'),
                              's': pd.Series([], dtype='float64'), 't': pd.Series([], dtype='float64'),
                              'u': pd.Series([], dtype='float64'), 'v': pd.Series([], dtype='float64'),
                              'w': pd.Series([], dtype='float64'), 'x': pd.Series([], dtype='float64'),
                              'y': pd.Series([], dtype='float64'), 'z': pd.Series([], dtype='int32'),
                              'aa': pd.Series([], dtype='bool'), 'ab': pd.Series([], dtype='int32'),
                              'ac': pd.Series([], dtype='float64'), 'ad': pd.Series([], dtype='float64'),
                              'ae': pd.Series([], dtype='int32'), 'af': pd.Series([], dtype='float64'),
                              'ag': pd.Series([], dtype='float64')})

data_type_dict = dummy_dataset.dtypes.apply(lambda x: x.name).to_dict()
print(dummy_dataset.dtypes)
for column, data_type in data_type_dict.items():
    # print(column, ": ",data_type)
    if data_type in ('int64', 'int32'):
        dummy_dataset[column] = random_integer()
    elif data_type == 'float64':
        dummy_dataset[column] = random_float()
    elif data_type == 'datetime64[ns]':
        dummy_dataset[column] = random_dates(start=pd.to_datetime('01-01-1940'), end=pd.to_datetime('01-01-2020'))
        dummy_dataset[column] = dummy_dataset[column].dt.strftime('%d-%m-%Y')
    elif data_type == 'object':
        dummy_dataset[column] = random_names('first_names')
    elif data_type == 'bool':
        dummy_dataset[column] = random_bool()

dummy_dataset.to_csv('dummy_data.csv', index=False)
