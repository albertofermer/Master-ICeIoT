from typing import Callable, List, cast, Dict, Iterable, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import signac
from signac import Project
from signac.contrib.job import Job
from signac_dashboard import Dashboard
from signac_dashboard.modules import TextDisplay#, PlotViewer
import pandas as pd
try:
    import markdown
    MARKDOWN = True
except ImportError:
    MARKDOWN = False
from flask import render_template, Markup # type: ignore


CLASSIFIER_TYPE = {
    'nominal': 'nominal',
    'ordinal_ecoc': 'ordinal (ECOC)',
    'ordinal_unimodal_binomial_ce': 'ordinal (Unimodal Binomial CE)',
}

DATASET = {
    'data/smear/30holdout_80_10_10.csv': 'smear',
    'data/retinopathy/30holdout_80_10_10_full.csv': 'retinopathy',
    'data/adience/30holdout_80_10_10.csv': 'adience',
    'data/csawm/30holdout_80_10_10.csv': 'csawm',
    'data/cbis-ddsm/30holdout_80_10_10.csv': 'cbis-ddsm',
}

EXPLANATION = {
    'gradcam': 'GradCAM',
    'iba': 'IBA',
}

class MyDashboard(Dashboard):
    def job_title(self, job):
        classifier_type = CLASSIFIER_TYPE[job.sp.classifier_type]
        dataset = DATASET[job.sp.dataset_csv_path]
        explanation = EXPLANATION[job.sp.explanation_method]

        return f'{dataset.capitalize()} dataset, {classifier_type} classifier, {explanation} explanation, partition {job.sp.partition}'


class MyTextDisplay(TextDisplay):
    def __init__(self, message: Callable[[Union[Job, Project]], Iterable[Union[str, Tuple[str, str]]]], necessary_key: Callable[[Union[Job, Project]], bool]=lambda _: True, *args, **kwargs):
        super(MyTextDisplay, self).__init__(*args, **kwargs)
        self.necessary_key = necessary_key
        self.message = message

    def get_cards(self, job_or_project: Union[Job, Project]):
        if not self.necessary_key(job_or_project):
            return []

        def apply_markdown(msg):
            if self.markdown:
                if MARKDOWN:
                    return Markup(markdown.markdown(
                        msg, extensions=[
                            'markdown.extensions.attr_list',
                            'markdown.extensions.tables',
                    ]))
                else:
                    return "Error: Install the 'markdown' library to render Markdown."
            else:
                return msg

        return [{
            "name": title_and_or_message[0] if isinstance(title_and_or_message, tuple) else self.name,
            "content": render_template(self.template, msg=apply_markdown(title_and_or_message[1] if isinstance(title_and_or_message, tuple) else title_and_or_message)),
        } for title_and_or_message in self.message(job_or_project)]

        msg = self.message(job_or_project)
        if self.markdown:
            if MARKDOWN:
                msg = Markup(markdown.markdown(
                    msg, extensions=[
                        'markdown.extensions.attr_list',
                        'markdown.extensions.tables',
                    ]))
            else:
                msg = ("Error: Install the 'markdown' library to render "
                       "Markdown.")
        return [{'name': self.name,
                 'content': render_template(self.template, msg=msg)}]


def normalize_curve(ys: List[float]):
    npys = np.array(ys)
    first = npys[0]
    last = npys[-1]
    normalized_npys = (npys - last) / (first - last)
    return normalized_npys

def rf_curves_plotly_data(morf_curve: List[float], lerf_curve: List[float]):
    x = np.linspace(0, 100, len(morf_curve)).tolist()
    normalized_morf_curve = normalize_curve(morf_curve)
    normalized_lerf_curve = normalize_curve(lerf_curve)

    df = pd.DataFrame({'morf_curve': normalized_morf_curve, 'lerf_curve': normalized_lerf_curve}, index=x)
    
    # split data into chunks where averages cross each other
    df['label'] = np.where(df['lerf_curve'] >= df['morf_curve'], 1, 0)
    df['group'] = df['label'].ne(df['label'].shift()).cumsum()
    dfs = [data for _, data in df.groupby('group')]

    # custom function to set fill color
    def fillcol(label):
        if label >= 1:
            return 'rgba(0,250,0,0.3)'
        else:
            return 'rgba(250,0,0,0.3)'

    
    traces = list()
    for prevdf, df, nextdf in zip([None]+dfs[:-1], dfs, dfs[1:]+[None]):
        if (prevdf is not None) and (not prevdf.empty):
            x1, x2 = cast(float, prevdf.index[-1]), cast(float, df.index[0])
            p1, p2 = prevdf['morf_curve'].iloc[-1], df['morf_curve'].iloc[0]
            q1, q2 = prevdf['lerf_curve'].iloc[-1], df['lerf_curve'].iloc[0]
            m1 = (p2 - p1) / (x2 - x1)
            m2 = (q2 - q1) / (x2 - x1)
            b1 = p1 - m1 * x1
            b2 = q1 - m2 * x1
            xprev = (b2 - b1) / (m1 - m2)
            yprev = m1 * xprev + b1
        else:
            xprev = None
            yprev = None

        if (nextdf is not None) and (not nextdf.empty):
            x1, x2 = cast(float, df.index[-1]), cast(float, nextdf.index[0])
            p1, p2 = df['morf_curve'].iloc[-1], nextdf['morf_curve'].iloc[0]
            q1, q2 = df['lerf_curve'].iloc[-1], nextdf['lerf_curve'].iloc[0]
            m1 = (p2 - p1) / (x2 - x1)
            m2 = (q2 - q1) / (x2 - x1)
            b1 = p1 - m1 * x1
            b2 = q1 - m2 * x1
            xnext = (b2 - b1) / (m1 - m2)
            ynext = m1 * xnext + b1
        else:
            xnext = None
            ynext = None

        xs = ([xprev] if xprev is not None else []) + df.index.to_list() + ([xnext] if xnext is not None else [])
        morf = ([yprev] if yprev is not None else []) + df['morf_curve'].to_list() + ([ynext] if ynext is not None else [])
        lerf = ([yprev] if yprev is not None else []) + df['lerf_curve'].to_list() + ([ynext] if ynext is not None else [])

        traces.extend([
            {
                "showlegend": False,
                "hoverinfo": "skip",
                "x": xs,
                "y": morf,
                "line": {"color": "rgba(0,0,0,0)"},
            },
            {
                "showlegend": False,
                "hoverinfo": "skip",
                "x": xs,
                "y": lerf,
                "line": {"color": "rgba(0,0,0,0)"},
                "fill": "tonexty",
                "fillcolor": fillcol(df['label'].iloc[0]),
            },
        ])

    traces.extend([{
            "x": x,
            "y": normalized_morf_curve.tolist(),
            "name": "MoRF",
            "line": {"color": "#1f77b4"},
        },
        {
            "x": x,
            "y": normalized_lerf_curve.tolist(),
            "name": "LeRF",
            "line": {"color": "#ff7f0e"},
        }
    ])

    return traces

def average_curves(job_list: List[Job]) -> List[Dict]:
    average_morf_curve = np.array([j.doc['explanation_results']['morf_curve'] for j in job_list if 'explanation_results' in j.doc]).mean(axis=0).tolist()
    average_lerf_curve = np.array([j.doc['explanation_results']['lerf_curve'] for j in job_list if 'explanation_results' in j.doc]).mean(axis=0).tolist()
    return rf_curves_plotly_data(average_morf_curve, average_lerf_curve)


def project_plotly_args(project: signac.Project) -> Iterable[Tuple[str, List[Dict], Dict]]:
    base_layout = {"xaxis": {"title": "degradation of x (%)"}, "yaxis": {"title": "normalized score"}}
    for (dataset_csv_path, explanation_method, classifier_type, cam_layer), jobs in project.groupby(
        ['dataset_csv_path', 'explanation_method', 'classifier_type', 'cam_layer']
    ):
        p_classifier_type = CLASSIFIER_TYPE[classifier_type]
        p_dataset = DATASET[dataset_csv_path]
        p_explanation = EXPLANATION[explanation_method]
        job_list = list(jobs)
        n_available = sum(1 for j in job_list if 'explanation_results' in j.doc)
        if n_available == 0:
            continue
        title = f'MoRF vs LeRF: {p_dataset.capitalize()} dataset, {p_classifier_type} classifier, {p_explanation} explanation (avg over {n_available})'
        yield title, average_curves(job_list), base_layout

def project_test_results(project: signac.Project) -> Iterable[Tuple[str, str]]:
    for (dataset_csv_path, explanation_method, classifier_type, cam_layer), jobs in project.groupby(
        ['dataset_csv_path', 'explanation_method', 'classifier_type', 'cam_layer']
    ):
        p_classifier_type = CLASSIFIER_TYPE[classifier_type]
        p_dataset = DATASET[dataset_csv_path]
        p_explanation = EXPLANATION[explanation_method]
        job_list = list(jobs)
        n_available = sum(1 for j in job_list if 'test_results' in j.doc)
        if n_available == 0:
            continue
        title = f'Results: {p_dataset.capitalize()} dataset, {p_classifier_type} classifier, {p_explanation} explanation (avg over {n_available})'

        yield title, results_function(job_list[0])[0]


def per_class_results(key):
    def per_class_results(job):
        as_series = pd.DataFrame(dict(job.doc[key]['result_metrics_per_class']))
        str_series = as_series.apply(lambda s: s.map(lambda n: f'{n:.04f}'))
        return [str_series.to_markdown(disable_numparse=True, colalign=('left',) + ('right',)*len(as_series.columns))]
    return per_class_results


def results_function(j):
    as_series = pd.Series(j.doc['test_results']['result_metrics'], name='Test value')
    str_series = as_series.map(lambda n: f'{n:.04f}')
    return [str_series.to_markdown(disable_numparse=True, colalign=('left', 'right'))]


def explanation_results_function(j):
    as_series = pd.Series({k: v for k, v in j.doc['explanation_results'].items() if pd.api.types.is_number(v)})


#class MyPlotViewer(PlotViewer):
class MyPlotViewer:
    def __init__(self, necessary_key, name="Plot Viewer", plotly_args: Callable[[Union[Job, Project]], Iterable[Tuple[str, List[Dict], Dict]]] = ..., context="JobContext", template="cards/plot_viewer.html", **kwargs):
        self.necessary_key = necessary_key
        super().__init__(name, plotly_args, context, template, **kwargs)

    def get_cards(self, job_or_project):
        if not self.necessary_key(job_or_project):
            return []
        return super().get_cards(job_or_project)


def main():
    MyDashboard(modules=[
        MyTextDisplay(name='Results',
                      necessary_key=lambda j: 'test_results' in j.doc,
                      message=results_function,
                      markdown=True),
        MyTextDisplay(name='Results per-class',
                      necessary_key=lambda j: 'test_results' in j.doc,
                      message=per_class_results('test_results'),
                      markdown=True),
        # MyTextDisplay(name='Explanation results',
        #               necessary_key=lambda j: 'explanation_results' in j.doc,
        #               message=explanation_results_function,
        #               markdown=True),
        # # MyPlotViewer(
        #     name="MoRF vs LeRF curve",
        #     necessary_key=lambda j: ('explanation_results' in j.doc) and ('morf_curve' in j.doc['explanation_results']) and ('lerf_curve' in j.doc['explanation_results']),
        #     plotly_args=lambda j: zip([''],
        #         [rf_curves_plotly_data(list(j.doc['explanation_results']['morf_curve']), list(j.doc['explanation_results']['lerf_curve']))],
        #         [{"xaxis": {"title": "degradation of x (%)"}, "yaxis": {"title": "normalized score"}}],
        #     )
        # ),
        # PlotViewer(
        #     context='ProjectContext',
        #     name="Average MoRF vs LeRF curve",
        #     plotly_args=lambda p: project_plotly_args(cast(Project, p)),
        # ),
        # MyTextDisplay(name='Explanation results',
        #               message=lambda p: project_test_results(cast(Project, p)),
        #               markdown=True),
    ]).main()


if __name__ == '__main__':
    main()