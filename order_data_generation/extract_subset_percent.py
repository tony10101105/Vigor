'''
This script extract a subset of training data.
Eg., 10% data: nr3d_train_LLM_step4_485.csv -> nr3d_train_LLM_step4_485_0.1.csv
'''


import math
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--percent", default=0.1, help='percent of data extracted')
args = parser.parse_args()
assert args.percent <= 1


train_data_path = 'nr3d_train_LLM_step4_485.csv'
out_train_path = 'nr3d_train_LLM_step4_485_{}.csv'.format(args.percent)

train_df = pd.read_csv(train_data_path)
train_df = train_df.sample(frac=1, random_state=2023)

new_train_df = []
data_num = math.floor(args.percent * train_df.shape[0])
print('data num: ', data_num)
for i in range(data_num):
    new_train_df.append(train_df.iloc[i].to_list())

new_train_df  = pd.DataFrame(new_train_df, columns=train_df.columns)
new_train_df.to_csv(out_train_path)