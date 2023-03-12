```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

Load data from file. Here we are only interested in accepted solutions, so we filter out the others.


```python
# read data
data = pd.read_csv('../data/processed/full-data.csv')

# keep only accepted solutions
data = data[data.result == 0]
```

Define important descriptors of the dataset.


```python
meta_features = [
    'problem_id', 'submission_id', 'language', 'result'
]
features = [
    'connected_components',
    'loop_statements',
    'conditional_statements',
    'jump_targets',
    'conditionals',
    'cycles',
    'paths',
    'cyclomatic_complexity',
    'avg_node_degree',
    'avg_clust_coeff_node',
    'avg_avg_degree_node_neighbors',
    'avg_avg_clust_coeff_node_neighbors',
    'avg_edges_node_out_egonet',
    'avg_neighbors_node_out_egonet',
    'avg_outgoing_edges_node_out_egonet',
    'avg_edges_node_in_egonet',
    'avg_neighbors_node_in_egonet',
    'avg_outgoing_edges_node_in_egonet',
    'eg1', 'eg2', 'eg3', 'eg4', 'eg5',
    'eg6', 'eg7', 'eg8', 'eg9', 'eg10'
]

# problem IDs
problem_ids = [
    6, 16, 18, 19, 21, 22, 23, 34, 35, 39, 42, 43, 45, 48, 53, 56
]

# languages
languages = ['C', 'CPP', 'JAVA', 'PYTHON']
```

Scale data.


```python
# scale data
ss = StandardScaler()
data[features] = ss.fit_transform(data[features])
```

Select example for demonstration. For instance, solutions to problem 21 written in Python.


```python
# filter for problem ID and language, e.g., problem 21 and Python
sample_data = data[(data.problem_id == 21) & (data.language == 'PYTHON')]
```

# Feature Selection
As there are too many features, we need to figure out which can be eliminated without sacrificing accuracy. To this end, we can use the Pearson's correlation coefficient to identify pairs of correlated features which, thus, do not need to be considered simultaneously. The correlation coefficient has values between -1 to 1: a value closer to 0 implies weaker correlation (exact 0 implies no correlation); a value closer to 1 implies stronger positive correlation; and a value closer to -1 implies stronger negative correlation.


```python
# create dataframe
sample_df = pd.DataFrame(data)

# correlation matrix
corr_mtx = sample_df[features].corr()

# absolute correlation matrix
corr_mtx_abs = corr_mtx.abs()
```

We can plot the correlation matrix as a heatmap to have a visual idea of the correlated features. As for any pair of features, the correlation coefficient  of `(X, Y)` is the same as `(Y, X)`, we can omit the upper triangle. Furthermore, for any feature `X`, `(X, X)` is always 1, so it may be omitted as well.


```python
# create plot placeholder
f, ax = plt.subplots(figsize=(12, 10))

# mask to get upper triangle of a matrix
mask = np.triu(np.ones_like(corr_mtx_abs, dtype=bool))

# configure a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# draw the heatmap
sns.heatmap(corr_mtx_abs, annot=True, mask=mask, cmap=cmap)
```




    <AxesSubplot:>




    
![png](feature-selection_files/feature-selection_12_1.png)
    


Through the above visualization, a few high correlation values are noticeable. To exclude features highly correlated
with other features in the selected set, we must decide on an adequate correlation threshold, i.e., the value above
which a pair is considered identical. Correlation thresholds are typically values equal or above 0.8 (e.g., 0.85, 0
.90, and 0.95). In this case, as there are too many features and mostly low correlation values, 0.85 was the value
selected.


```python
corr_threshold = 0.9
```

Then, we just iterate through the upper or bottom triangle of the matrix and identify pairs (row, column) whose correlation value is above the selected threshold.


```python
corr_pairs = []
for i in range(len(corr_mtx_abs.columns)):
    for j in range(i):
        if abs(corr_mtx_abs.iloc[i, j]) > corr_threshold:
            rowname = corr_mtx_abs.columns[i]
            colname = corr_mtx_abs.columns[j]
            corr_pairs.append((rowname, colname))
print('Nr. of correlation pairs found: {}'.format(len(corr_pairs)))
```

    Nr. of correlation pairs found: 48


Now, for each pair, one of the elements is dropped, if both are still in the selected feature set.


```python
drop_features = []
for (rowname, colname) in corr_pairs:
    if not drop_features.__contains__(rowname) and not drop_features.__contains__(colname):
        drop_features.append(rowname)
sample_df_clean = sample_df.drop(columns=drop_features, inplace=False)
print(sample_df_clean.columns)
```

    Index(['problem_id', 'submission_id', 'language', 'result',
           'connected_components', 'loop_statements', 'conditional_statements',
           'jump_targets', 'conditionals', 'cycles', 'paths', 'avg_node_degree',
           'avg_clust_coeff_node', 'avg_avg_degree_node_neighbors',
           'avg_neighbors_node_out_egonet', 'eg1', 'eg2'],
          dtype='object')


Actually, our dataset is not meant to be used as a whole, but rather the sub-datasets whose records have the same pair `
(problem ID, language)` separately. In this case, we should identify the correlated features in each of these
sub-datasets also separately. Therefore, we have done a poll considering the votes of each of these sub-datasets,
whose size is greater or equal to 10 records, regarding the features to drop. Those with 1/3 or more votes are dropped.


```python
drop_voting = {feat: 0 for feat in features}
count_voters = 0
for problem_id in problem_ids:
    for language in languages:
        data_local = data[(data.problem_id == problem_id) & (data.language == language)]
        if len(data_local.index) < 10:
            continue # no vote for smaller datasets
        count_voters = count_voters + 1
        df_local = pd.DataFrame(data_local)
        corr_mtx = df_local[features].corr()
        corr_mtx_abs = corr_mtx.abs()
        corr_pairs = []
        for i in range(len(corr_mtx_abs.columns)):
            for j in range(i):
                if abs(corr_mtx_abs.iloc[i, j]) > corr_threshold:
                    rowname = corr_mtx_abs.columns[i]
                    colname = corr_mtx_abs.columns[j]
                    corr_pairs.append((rowname, colname))
        drop_features = []
        for (rowname, colname) in corr_pairs:
            if not drop_features.__contains__(rowname) and not drop_features.__contains__(colname):
                drop_voting[rowname] = drop_voting[rowname] + 1
                drop_features.append(rowname)

drop_features = list(dict(filter(lambda it: it[1] >= count_voters / 3, drop_voting.items())).keys())
print(drop_features)
```

    ['jump_targets', 'conditionals', 'paths', 'cyclomatic_complexity', 'avg_avg_clust_coeff_node_neighbors', 'avg_edges_node_out_egonet', 'avg_neighbors_node_out_egonet', 'avg_outgoing_edges_node_out_egonet', 'avg_edges_node_in_egonet', 'avg_neighbors_node_in_egonet', 'avg_outgoing_edges_node_in_egonet', 'eg4', 'eg5', 'eg6', 'eg7', 'eg8', 'eg9', 'eg10']


The set of features to use is then


```python
selected_features = [feat for feat in features if feat not in drop_features]
print(selected_features)
```

    ['connected_components', 'loop_statements', 'conditional_statements', 'cycles', 'avg_node_degree', 'avg_clust_coeff_node', 'avg_avg_degree_node_neighbors', 'eg1', 'eg2', 'eg3']

