from collections import defaultdict
from htcondor.htcondor import JobEventLog, JobEventType, Schedd
import signac
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.formula.api import ols
import numpy as np


dataset_csv_paths = {
    'smear': 'data/smear/30holdout_80_10_10.csv',
    'retinopathy': 'data/retinopathy/30holdout_80_10_10.csv', 
    'adience': 'data/adience/30holdout_80_10_10.csv',
}
operation = 'evaluate_trained_model'

def main():
    project = signac.get_project()
    schedd = Schedd()
    dfs = dict()
    for dataset_name, dataset_csv_path in dataset_csv_paths.items():
        print(dataset_name)
        rows = defaultdict(lambda: list())
        for job in project.find_jobs({'dataset_csv_path': dataset_csv_path}):
            events = list(JobEventLog(job.fn(f'{operation}/log')).events(0))
            
            job_terminated_events = [e for e in events if e.type == JobEventType.JOB_TERMINATED]
            if (not job_terminated_events) or (job_terminated_events[-1]['ReturnValue'] != 0):
                print(f'Error in job {job.id} with statepoint: {job.sp}')
                continue
            job_terminated_event = job_terminated_events[-1]

            rows['id'].append(job.id)
            rows['batch_size'].append(job.sp.batch_size)
            rows['n_workers'].append(job.sp.n_workers)
            rows['cluster_id'].append(job_terminated_event.cluster)
            rows['proc_id'].append(job_terminated_event.proc)
            rows[f'event_{operation}_memory_usage_mb'].append(job_terminated_event['MemoryUsage'])
            rows['peak_train_cuda_memory_usage_mb'].append(job.doc.get('train_max_cuda_memory_usage', np.nan) / (1024 * 1024))
            rows['peak_test_cuda_memory_usage_mb'].append(job.doc.get('test_max_cuda_memory_usage', np.nan) / (1024 * 1024))
            
        df = pd.DataFrame(rows).sort_values(by=['batch_size', 'n_workers'])

        jobs_history = list(schedd.history(
            ' || '.join(f'((ClusterId == {row.cluster_id}) && (ProcId == {row.proc_id}))' for row in df.itertuples()),
            ['MemoryUsage', 'ClusterId', 'ProcId'])
        )
        job_memory_usage = {(h['ClusterId'], h['ProcId']): h['MemoryUsage'].eval() for h in jobs_history}
        df[f'peak_{operation}_memory_usage_mb'] = [job_memory_usage.get((row.cluster_id, row.proc_id), np.nan) for row in df.itertuples()]
        dfs[dataset_name] = df

        # data_pipeline = make_pipeline(StandardScaler(), LinearRegression())
        # data_pipeline.fit(df[['batch_size', 'n_workers']].to_numpy(), df['peak_memory_usage_mb'].to_numpy())
        # df['estimated_peak_memory_usage_mb'] = data_pipeline.predict(df[['batch_size', 'n_workers']].to_numpy())
        # print(df)
        # print(f"score: {data_pipeline.score(df[['batch_size', 'n_workers']].to_numpy(), df['peak_memory_usage_mb'].to_numpy())}")

        # model = ols('memory_usage_mb ~ n_workers + batch_size + n_workers * batch_size', data=df).fit()
        # print(model.summary())
    
    with pd.ExcelWriter("memory_stats_prueba.xlsx") as writer:
        for dataset_name, df in dfs.items():
            df.to_excel(writer, sheet_name=dataset_name, index=False)



if __name__ == '__main__':
    main()