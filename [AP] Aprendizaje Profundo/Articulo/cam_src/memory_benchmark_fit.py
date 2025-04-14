from collections import defaultdict
from htcondor.htcondor import JobEventLog, JobEventType, Schedd
import signac
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.formula.api import ols
from itertools import product
import numpy as np


dataset_csv_paths = {
    'smear': 'data/smear/30holdout_80_10_10.csv',
    'retinopathy': 'data/retinopathy/30holdout_80_10_10.csv', 
    'adience': 'data/adience/30holdout_80_10_10.csv',
}

stats_files = ['memory_stats1.xlsx', 'memory_stats2.xlsx']

models = {
    'event_memory_usage_mb': ['n_workers'],
    'peak_cuda_memory_usage_mb': ['batch_size'],
}

def main():
    dataset_stats = {dataset_name: [pd.read_excel(file, sheet_name=dataset_name) for file in stats_files] for dataset_name in dataset_csv_paths}
    for dataset_name, stat_dfs in dataset_stats.items():
        print(dataset_name)

        combined_df = stat_dfs[0].copy().drop(columns=['cluster_id', 'proc_id'])
        for c in set(combined_df.columns) - {'id', 'batch_size', 'n_workers'}:
            for batch_size, n_workers in product(combined_df['batch_size'].unique(), combined_df['n_workers'].unique()):
                values = [df.loc[(df['batch_size'] == batch_size) & (df['n_workers'] == n_workers), c].item() for df in stat_dfs]
                combined_df.loc[(combined_df['batch_size'] == batch_size) & (combined_df['n_workers'] == n_workers), c] = max(values, key=lambda n: -np.inf if np.isnan(n) else n)

        for dependent_variable, independent_variables in models.items():
            print(f'Explaining {dependent_variable} with: {independent_variables}')
            data_pipeline = make_pipeline(StandardScaler(), LinearRegression())
            data_pipeline.fit(combined_df[independent_variables].to_numpy(), combined_df[dependent_variable].to_numpy())
            combined_df[f'estimated_{dependent_variable}'] = data_pipeline.predict(combined_df[independent_variables].to_numpy())
            print('Coefficients:')
            for i, variable in enumerate(independent_variables):
                print(f"- {variable}: {data_pipeline['linearregression'].coef_[i]}")
            print(f"Intercept: {data_pipeline['linearregression'].intercept_}")
            print(f"score: {data_pipeline.score(combined_df[independent_variables].to_numpy(), combined_df[dependent_variable].to_numpy())}")
            print('')

        print(combined_df)
        # model = ols('memory_usage_mb ~ n_workers + batch_size + n_workers * batch_size', data=df).fit()
        # print(model.summary())



if __name__ == '__main__':
    main()