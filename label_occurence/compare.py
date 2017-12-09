import pandas as pd
import numpy as np
pd.options.display.max_rows = 999
result = pd.read_csv('result.csv')
training = pd.read_csv('train_binary_Y.csv')
result.columns = ['label', 'occ_result']
training.columns = ['label', 'occ_training']
result['perc_result']= result['occ_result']/result['occ_result'].sum()
training['perc_training']= training['occ_training']/training['occ_training'].sum()
df = result.merge(training, how='outer', left_on='label', right_on='label')
df = df.sort_values('occ_training', ascending=False)
print(df[['label','perc_result','perc_training']])
