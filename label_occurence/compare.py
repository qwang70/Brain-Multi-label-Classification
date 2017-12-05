import pandas as pd
import numpy as np

result = pd.read_csv('result.csv')
training = pd.read_csv('train_binary_Y.csv')
result.columns = ['label', 'occ_result']
training.columns = ['label', 'occ_training']
df = result.merge(training, how='outer', left_on='label', right_on='label')
print(df)
