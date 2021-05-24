
from scipy.stats import mannwhitneyu
import pandas as pd


def find_distribution(data1, data2):
    stat, p = mannwhitneyu(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

#data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
#data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]

test_df=pd.read_csv("sample_dataset.csv")

columns=['State', 'Account length', 'Area code', 'International plan', 'Voice mail plan', 'Number vmail messages', 'Total day minutes', 'Total day calls', 'Total day charge', 'Total eve minutes', 'Total eve calls', 'Total eve charge', 'Total night minutes', 'Total night calls', 'Total night charge', 'Total intl minutes', 'Total intl calls', 'Total intl charge', 'Customer service calls', 'Churn']


#sample_df=pd.DataFrame(test_df[["Total day calls", "Total eve calls"]])

col1=test_df["Total day calls"].to_list()
col2= test_df["Total eve calls"].to_list()

find_distribution(col1, col2)






