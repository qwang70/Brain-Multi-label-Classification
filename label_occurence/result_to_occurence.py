import numpy as np
import pandas as pd

#fn = 'train_binary_Y'
fn = 'result'
#fn = 'post_result'
result = np.load("../{}.npy".format(fn))
d = {}
for row in result:
    label =  ' '.join((str(e) for e in np.nonzero(row)[0]))
    if label in d:
        d[label] += 1
    else:
        d[label] = 1
df = pd.DataFrame.from_dict(data=d,orient='index')
df = df.sort_values([0], ascending=False)
df.to_csv("{}.csv".format(fn))
