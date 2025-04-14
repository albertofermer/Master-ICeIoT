import argparse
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import signac
from flow import FlowProject
from signac.contrib.filterparse import parse_filter_arg


def add_filters_to_parser(parser: argparse.ArgumentParser):
    FlowProject._add_job_selection_args(parser)


def get_filters_from_args(args: List[str]):
    filter = parse_filter_arg(args)
    return filter if filter is not None else dict()

base_model_attributes = ['seed', 'dataset_csv_path', 'partition', 'base_model', 'classifier_type', 'learning_rate', 'patience', 'max_epochs', 'batch_size']

def main():
    project = signac.get_project()

    parser = argparse.ArgumentParser()
    add_filters_to_parser(parser)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    sp_filter = get_filters_from_args(args.filter)
    doc_filter = get_filters_from_args(args.doc_filter)

    df_dict = defaultdict(lambda: list())
    for attrs, jobs in  project.find_jobs(sp_filter, doc_filter).groupby(base_model_attributes):
        job = next(iter(jobs))
        if 'test_results' not in job.doc:
            print(f'Skipping job {job.id}')
            continue

        df_dict['statepoint'].append(str(dict(job.sp)))
        ds_name = Path(job.sp.dataset_csv_path).parts[1]
        df_dict['dataset'].append(ds_name)
        df_dict['base_model'].append(job.sp.base_model)
        df_dict['classifier_type'].append(job.sp.classifier_type)
        df_dict['partition'].append(job.sp.partition)
        for metric in job.doc['test_results']['result_metrics']:
            df_dict[metric].append(job.doc['test_results']['result_metrics'][metric])

    df = pd.DataFrame(df_dict).sort_values(['dataset', 'base_model', 'classifier_type', 'partition'])
    average_df = df.groupby(['dataset', 'base_model', 'classifier_type']).mean(numeric_only=True).drop(columns='partition')
    with pd.ExcelWriter(str(args.output_dir / 'results.xlsx')) as writer:
        df.to_excel(writer, 'all')
        average_df.to_excel(writer, 'average')

    # for v, p in zip((True, False), ('withoutliers', 'withoutoutliers')):
    #     fig, ax = plt.subplots(1, 1)
    #     sns.boxplot(df, x='dataset', y='average_drop', hue='classifier_type', ax=ax, showfliers=v)
    #     fig.savefig(str(args.output_dir / f'results_average_drop_{p}.png'))

    #     fig, ax = plt.subplots(1, 1)
    #     sns.boxplot(df, x='dataset', y='confidence_increase', hue='classifier_type', ax=ax, showfliers=v)
    #     fig.savefig(str(args.output_dir / f'results_confidence_increase_{p}.png'))




if __name__ == '__main__':
    main()
