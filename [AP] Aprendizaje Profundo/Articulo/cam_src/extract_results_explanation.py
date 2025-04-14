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
from tqdm import tqdm


def add_filters_to_parser(parser: argparse.ArgumentParser):
    FlowProject._add_job_selection_args(parser)


def get_filters_from_args(args: List[str]):
    filter = parse_filter_arg(args)
    return filter if filter is not None else dict()


def main():
    project = signac.get_project()

    parser = argparse.ArgumentParser()
    add_filters_to_parser(parser)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    sp_filter = get_filters_from_args(args.filter)
    doc_filter = get_filters_from_args(args.doc_filter)

    df_dict = defaultdict(lambda: list())
    total_jobs = 0
    skipped_jobs = 0
    for job in  tqdm(project.find_jobs(sp_filter, doc_filter)):
        total_jobs +=1

        if ('explanation_results' not in job.doc):
            print(f'Skipping job {job.id}')
            skipped_jobs += 1
            continue

        ds_name = Path(job.sp.dataset_csv_path).parts[1]
        df_dict['dataset'].append(ds_name)
        df_dict['classifier_type'].append(job.sp.classifier_type)
        df_dict['explanation_method'].append(job.sp.explanation_method)
        df_dict['step_exponent'].append(job.sp.get('step_exponent', 'na'))
        df_dict['cam_layer'].append(job.sp.cam_layer)
        df_dict['partition'].append(job.sp.partition)
        df_dict['average_drop'].append(-job.doc['explanation_results']['average_drop'])
        df_dict['confidence_increase'].append(job.doc['explanation_results']['confidence_increase'])
        for metric in job.doc['explanation_results']['degradations']:
            df_dict[f'{metric}_degradation'].append(job.doc['explanation_results']['degradations'][metric])
        df_dict['valid_samples_ratio'].append(job.doc['explanation_results']['valid_samples_ratio'])

    print(f'Skipped a total of {skipped_jobs}/{total_jobs} jobs ({(skipped_jobs / total_jobs)*100:.1f}%)')

    df = pd.DataFrame(df_dict).sort_values(['dataset', 'classifier_type', 'explanation_method', 'step_exponent', 'cam_layer', 'partition'])
    df['count'] = 1
    df_groups = df.groupby(['dataset', 'classifier_type', 'explanation_method', 'step_exponent', 'cam_layer'])
    average_df = df_groups.mean().drop(columns=['partition', 'count'])
    sum_df = df_groups.sum()
    average_df['count'] = sum_df['count']
    with pd.ExcelWriter(str(args.output_dir / 'results.xlsx')) as writer:
        df.drop(columns='count').to_excel(writer, 'all')
        average_df.to_excel(writer, 'average')

    df_toboxplot = df.copy()
    df_toboxplot['classtype_and_expmethod'] = df_toboxplot['classifier_type'] + df_toboxplot['explanation_method']

    def save_boxplot(path, dataset, x, y, hue):
        fig, ax = plt.subplots(1, 1)
        sns.boxplot(df_toboxplot.loc[df_toboxplot['dataset'] == dataset], x=x, y=y, hue=hue, ax=ax)
        fig.savefig(str(path.with_suffix('.png')))
        fig.clear()
        plt.close(fig)

    for dataset in df_toboxplot['dataset'].unique():
        dataset_dir = args.output_dir / dataset
        dataset_dir.mkdir(exist_ok=True)
        save_boxplot(dataset_dir / f'average_drop', dataset, 'explanation_method', 'average_drop', 'classifier_type')
        save_boxplot(dataset_dir / f'confidence_increase', dataset, 'explanation_method', 'confidence_increase', 'classifier_type')

        for c in filter(lambda name: name.endswith('_degradation'), df_toboxplot.columns):
            save_boxplot(dataset_dir / c, dataset, 'explanation_method', c, 'classifier_type')


if __name__ == '__main__':
    main()
