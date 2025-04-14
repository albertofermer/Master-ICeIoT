#!/home/javierbg/miniconda3/envs/cam/bin/python

import hashlib
import os
import sys
import threading
import time
import traceback
import warnings
from functools import partial, partialmethod
from pathlib import Path
from typing import List, Optional, Iterator, Tuple, Union, Sequence, cast, overload, Dict

import pytorch_lightning as pl
import torch
from torch import Tensor
import torchvision.models
import torch.utils.data as tdata
import torch.nn.functional as F
import numpy as np
import scipy
from flow import FlowProject, aggregator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_lite.utilities.warnings import PossibleUserWarning
from signac import H5Store
from signac.contrib.job import Job
from sklearn.metrics import confusion_matrix
from torchvision.models.quantization.utils import _replace_relu as replace_relu  # type: ignore
from torchvision.transforms.functional import resize, InterpolationMode
from tqdm import tqdm

import fix_cuda_device_count
from condor import CondorEnvironment
from data import (AdienceDataModule, ExperimentDataModule,
                  RetinopathyDataModule, SmearDataModule,
                  CSAWMDataModule, CBISDDSMDataModule)
from metrics import metrics
from model import AttributionModel, ExperimentModel, PredictOutput


tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line)) # type: ignore


warnings.showwarning = warn_with_traceback
warnings.filterwarnings("ignore", message="root_directory is deprecated", category=FutureWarning, module='flow.project')
warnings.filterwarnings("ignore", message=r"The dataloader, [^,]*, does not have many workers which may be a bottleneck", category=PossibleUserWarning, module='pytorch_lightning')


class AYRNACondorEnvironment(CondorEnvironment):
    hostname_pattern = r'srvrryc(arn|inf)\d{2}\.priv\.uco\.es$'
    template = 'custom-condor-submit.sh'

class Project(FlowProject):
    pass


full_experiment = Project.make_group(name='full_experiment')
training_operations = Project.make_group(name='training_operations')
evaluating_operations = Project.make_group(name='evaluating_operations')

base_model_attributes = ['seed', 'dataset_csv_path', 'partition', 'base_model', 'classifier_type', 'learning_rate', 'patience', 'max_epochs', 'batch_size']
model_aggregator = aggregator.groupby(base_model_attributes)#, select=lambda j: j.sp.partition >= 30)


def load_data(job: Job) -> ExperimentDataModule:
    if 'smear' in job.sp.dataset_csv_path:
        return SmearDataModule(job)
    elif 'retinopathy' in job.sp.dataset_csv_path:
        return RetinopathyDataModule(job)
    elif 'adience' in job.sp.dataset_csv_path:
        return AdienceDataModule(job)
    elif 'csawm' in job.sp.dataset_csv_path:
        return CSAWMDataModule(job)
    elif 'cbis-ddsm' in job.sp.dataset_csv_path:
        return CBISDDSMDataModule(job)
    else:
        raise NotImplementedError

BASE_MODEL_DIR = Path('models').resolve()

def build_model_checkpoint(job: Job) -> ModelCheckpoint:
    checkpoint_dirpath = BASE_MODEL_DIR / model_hash(job.sp) / 'checkpoints'
    # checkpoint_dirpath.mkdir(exist_ok=True, parents=True)
    return ModelCheckpoint(str(checkpoint_dirpath), monitor='val_loss', mode='min', save_last=True)


def last_checkpoint_path(mc: ModelCheckpoint) -> Optional[Path]:
    last_checkpoint_path = Path(f'{mc.dirpath}/{mc.CHECKPOINT_NAME_LAST}{mc.FILE_EXTENSION}')
    if last_checkpoint_path.is_file():
        return last_checkpoint_path
    else:
        return None

@Project.label
def model_trained(*jobs: Job) -> bool:
    h = model_hash(jobs[0].sp)
    assert all(model_hash(j.sp) == h for j in jobs[1:])
    checkpoint_callback = build_model_checkpoint(jobs[0])
    checkpoint_present = last_checkpoint_path(checkpoint_callback) is not None

    marked_as_finished = [('training_finished' in j.doc) and (j.doc['training_finished']) for j in jobs]

    if any(marked_as_finished) and (not checkpoint_present):
        for j in jobs:
            if 'training_max_cuda_memory_usage' in j.doc:
                del j.doc['training_max_cuda_memory_usage']
            if 'training_elapsed_seconds' in j.doc:
                del j.doc['training_elapsed_seconds']
            j.doc['training_finished'] = False
        return False

    if all(marked_as_finished) and checkpoint_present:
        return True

    if any(marked_as_finished) and checkpoint_present:
        marked_job = [j for f, j in zip(marked_as_finished, jobs) if f][0]
        for j in jobs:
            j.doc['training_finished'] = True
            j.doc['training_max_cuda_memory_usage'] = marked_job.doc['training_max_cuda_memory_usage']
            j.doc['training_elapsed_seconds'] = marked_job.doc['training_elapsed_seconds']
        return True
        
    return False



def results_saved(job):
    return ('test_results' in job.doc.keys()) and \
           ('confusion_matrix' in job.doc['test_results'].keys()) and \
           ('result_metrics' in job.doc['test_results'].keys()) and \
           ('result_metrics_per_class' in job.doc['test_results'].keys())


@FlowProject.label
def all_results_saved(*jobs):
    saved = [results_saved(j) for j in jobs]
    
    if all(saved):
        return True

    if any(saved):
        first_saved_job = [j for s, j in zip(saved, jobs) if s][0]
        for _, job in filter(lambda e: not e[0], zip(saved, jobs)):
            job.doc['test_results'] = first_saved_job.doc['test_results']
        return True

    return False


# MEMORY_USAGE = { # (n_workers_coefficient, intercept)
#     'data/smear/30holdout_80_10_10.csv': (8724.77, 28283.56),
#     'data/retinopathy/30holdout_80_10_10.csv': (7905.17, 29172.62), 
#     'data/adience/30holdout_80_10_10.csv': (8557.46, 28279.81),
# }

# CUDA_MEMORY_USAGE = { # (batch_size_coefficient, intercept)
#     'data/smear/30holdout_80_10_10.csv': (765.36, 1919.61),
#     'data/retinopathy/30holdout_80_10_10.csv': (766.52, 1920.78), 
#     'data/adience/30holdout_80_10_10.csv': (765.17, 1916.94),
# }

# def job_memory_requirement_gb(job):
#     n_workers_coeff, intercept = MEMORY_USAGE[job.sp.dataset_csv_path]
#     memory_mb = n_workers_coeff * job.sp.n_workers + intercept
#     return f'{(memory_mb / 1024) * 1.1:.1f}'


def model_hash(statepoint) -> str:
    config = tuple(sorted([(k, v) for k, v in statepoint.items() if k in base_model_attributes], key=lambda e: e[0]))
    md5hash = hashlib.md5()
    md5hash.update(bytes(str(config), 'utf-8'))
    return md5hash.hexdigest()


def train_cpu(*jobs):
    return jobs[0].sp.n_workers

def train_memory(*jobs):
    return int(2.65*1024)

@model_aggregator
@Project.operation(directives={'ngpu': 1, 'np': train_cpu, 'memory': train_memory})  # type: ignore
@Project.post(model_trained)  # type: ignore
def train_model(*jobs: Job):
    first_job = jobs[0]
    model_id = model_hash(first_job.sp)
    datamodule = load_data(first_job)
    checkpoint_callback = build_model_checkpoint(first_job)

    if (lcp := last_checkpoint_path(checkpoint_callback)) is not None:
        model = ExperimentModel.load_from_checkpoint(str(lcp), n_classes=datamodule.n_classes, job=first_job)
    else:
        model = ExperimentModel(datamodule.n_classes, first_job)

    trainer_callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=first_job.sp.patience),
        checkpoint_callback,
    ]
    logger = TensorBoardLogger(str(BASE_MODEL_DIR / model_id / 'logs'))
    logger.log_hyperparams({k: v for k, v in first_job.sp.items() if k in base_model_attributes})

    trainer = pl.Trainer(deterministic=True, max_epochs=first_job.sp.max_epochs, accelerator='gpu', devices=1,
                         log_every_n_steps=1, callbacks=trainer_callbacks, enable_progress_bar=False,
                         logger=logger)

    begin_time = time.perf_counter()
    trainer.fit(model=model, datamodule=datamodule)
    end_time = time.perf_counter()

    for job in jobs:
        job.doc['training_max_cuda_memory_usage'] = torch.cuda.max_memory_allocated('cuda:0')
        job.doc['training_elapsed_seconds'] = end_time - begin_time
        job.doc['training_finished'] = True



@model_aggregator
@Project.operation(directives={'ngpu': 1, 'np': train_cpu, 'memory': 12*1024})  # type: ignore
@Project.pre.after(train_model)  # type: ignore
@Project.post(all_results_saved)  # type: ignore
def evaluate_trained_model(*jobs: Job):
    first_job = jobs[0]
    datamodule = load_data(first_job)
    checkpoint_callback = build_model_checkpoint(first_job)

    assert (lcp := last_checkpoint_path(checkpoint_callback)) is not None
    model = ExperimentModel.load_from_checkpoint(str(lcp), n_classes=datamodule.n_classes, job=first_job)
    trainer = pl.Trainer(deterministic=True, accelerator='gpu', devices=1, enable_progress_bar=False)
    results = evaluate_model(trainer, model, datamodule)

    for job in jobs:
        job.doc['test_max_cuda_memory_usage'] = torch.cuda.max_memory_allocated('cuda:0')
        job.doc['test_results'] = results


def evaluate_model(trainer: pl.Trainer, module: pl.LightningModule, datamodule: ExperimentDataModule):
    predict_output: List[PredictOutput] = trainer.predict(module, datamodule=datamodule)  # type: ignore
    scores = torch.cat([p.score for p in predict_output])
    probas = torch.cat([p.proba for p in predict_output])
    targets = datamodule.predict_dataset.targets
    predicted = torch.argmax(scores, dim=1)

    cm = confusion_matrix(targets, predicted)
    em, empc = evaluation_metrics(targets, predicted, probas)
    return {
        'confusion_matrix': cm.tolist(),
        'result_metrics': em,
        'result_metrics_per_class': empc
    }


def evaluation_metrics(ytrue, ypred, probas):
    """
    Compute all the evaluation metrics into a dictionary.
    ytrue
        True labels
    ypred
        Predicted labels
    probas
        Predicted class probabilities
    Returns
    -------
    x:
        Dictionary containing all computed metrics (as listed in C{metrics.metric_list})
    """
    evaluated_metrics = dict()
    for metric_type in ('multinomial', 'ordinal'):
        for metric_name, m in metrics[metric_type]['labels'].items():
            evaluated_metrics[metric_name] = m(ytrue, ypred)
        for metric_name, m in metrics[metric_type]['scores'].items():
            evaluated_metrics[metric_name] = m(ytrue, probas)

    evaluated_metrics_per_class = dict()
    for metric_name, m in metrics['multinomial_per_class']['labels'].items():
        evaluated_metrics_per_class[metric_name] = m(ytrue, ypred)
    for metric_name, m in metrics['multinomial_per_class']['scores'].items():
        evaluated_metrics_per_class[metric_name] = m(ytrue, probas)
    return evaluated_metrics, evaluated_metrics_per_class


_TEST_SCORES_KEYS = [
    'targets',
    'original/scores', 'original/probas',
    'sharpened/scores', 'sharpened/probas',
] + \
[f'{order}_8x8/{occlusion_idx}/scores' for order in ('morf', 'lerf') for occlusion_idx in range(64)] + \
[f'{order}_8x8/{occlusion_idx}/probas' for order in ('morf', 'lerf') for occlusion_idx in range(64)]


def attribution_cpu(*jobs):
    return 1

def attribution_memory(*jobs):
    if 'adience' in jobs[0].sp.dataset_csv_path:
        return 3100
    elif 'cbis-ddsm' in jobs[0].sp.dataset_csv_path:
        return 1024
    elif 'retinopathy' in jobs[0].sp.dataset_csv_path:
        return 5120
    elif 'smear' in jobs[0].sp.dataset_csv_path:
        return 750
    else:
        raise RuntimeError


@Project.label
def test_outputs_computed(job: Job) -> bool:
    return job.doc.get('test_attribution_outputs_computed', False)

@Project.operation(directives={'ngpu': 1, 'np': attribution_cpu, 'memory': attribution_memory})  # type: ignore
@Project.pre.after(train_model)  # type: ignore
@Project.post(test_outputs_computed)  # type: ignore
def compute_test_attribution_outputs(job: Job):    
    datamodule = load_data(job)
    checkpoint_callback = build_model_checkpoint(job)

    assert (lcp := last_checkpoint_path(checkpoint_callback)) is not None
    model = cast(ExperimentModel, ExperimentModel.load_from_checkpoint(str(lcp), n_classes=datamodule.n_classes, job=job))
    replace_relu(model)

    trainer = pl.Trainer(deterministic='warn', accelerator='gpu', devices=1, enable_progress_bar=False)

    predict_output: List[PredictOutput] = trainer.predict(model, datamodule=datamodule)  # type: ignore
    scores_before = torch.cat([p.score for p in predict_output])
    probas_before = torch.cat([p.proba for p in predict_output])
    targets = torch.tensor(datamodule.predict_dataset.targets).long()

    trainer = pl.Trainer(deterministic='warn', accelerator='gpu', devices=1, enable_progress_bar=False,
                         inference_mode=False)
    model = model.to('cuda:0')

    def get_target_scores(datamodule: pl.LightningDataModule) -> Tuple[Tensor, Tensor]:
        predict_output: List[PredictOutput] = trainer.predict(model, datamodule)

        predict_output = cast(List[PredictOutput], predict_output)

        new_score = torch.cat([p.score for p in predict_output])
        new_proba = torch.cat([p.proba for p in predict_output])
        return new_score, new_proba

    attribution_model = AttributionModel(job, model, datamodule)
    explanations = torch.cat(trainer.predict(attribution_model, datamodule))  # type: ignore

    sharpened_dm = datamodule.masked_datamodule(explanations)

    scores_after_sharpening, probas_after_sharpening = get_target_scores(sharpened_dm)

    if 'attribution_scores' in job.stores:
        del job.stores['attribution_scores']
        
    with job.stores['attribution_scores'] as attribution_scores:
        attribution_scores = cast(H5Store, attribution_scores)
        attribution_scores['targets'] = targets.numpy()
        attribution_scores['original/scores'] = scores_before.numpy()
        attribution_scores['original/probas'] = probas_before.numpy()

        attribution_scores['sharpened/scores'] = scores_after_sharpening.numpy()
        attribution_scores['sharpened/probas'] = probas_after_sharpening.numpy()

    for order in ('morf', 'lerf'):
        occluded_datamodules = increasingly_occluded_datamodules(datamodule, explanations, reverse=(order == 'lerf'))

        for occlusion_idx, occluded_datamodule in enumerate(occluded_datamodules):
            occluded_scores, occluded_probas = get_target_scores(occluded_datamodule)

            with job.stores['attribution_scores'] as attribution_scores:
                attribution_scores[f'{order}_8x8/{occlusion_idx}/scores'] = occluded_scores.detach().numpy()
                attribution_scores[f'{order}_8x8/{occlusion_idx}/probas'] = occluded_probas.detach().numpy()
    

    job.doc['test_attribution_outputs_computed'] = True


@Project.label
def explanations_evaluated(job):
    return ('explanation_results' in job.doc) and \
        ('curves' in job.doc['explanation_results']) and \
        ('degradations' in job.doc['explanation_results'])

@Project.operation(directives={'np': 1, 'memory': 2*1024})  # type: ignore
@Project.pre.after(compute_test_attribution_outputs)  # type: ignore
@Project.post(explanations_evaluated)  # type: ignore
def evaluate_explanations(job):
    with job.stores['attribution_scores'] as attribution_scores:
        attribution_scores = cast(H5Store, attribution_scores)
        targets = np.array(attribution_scores['targets'])
        scores_before = np.array(attribution_scores['original/scores'])
        probas_before = np.array(attribution_scores['original/probas'])

        probas_after_sharpening = np.array(attribution_scores['sharpened/probas'])
    
    n_samples_original = targets.shape[0]
    target_scores_before = scores_before[np.arange(n_samples_original), targets]
    target_probas_before = probas_before[np.arange(n_samples_original), targets]

    valid_samples = target_probas_before != 0.0

    scores_before = scores_before[valid_samples]
    target_scores_before = target_scores_before[valid_samples]

    probas_before = probas_before[valid_samples]
    target_probas_before = target_probas_before[valid_samples]

    target_probas_after_sharpening = probas_after_sharpening[np.arange(n_samples_original), targets]
    target_probas_after_sharpening = target_probas_after_sharpening[valid_samples]

    average_drop = -((np.clip(target_probas_before - target_probas_after_sharpening, 0, None) / np.abs(target_probas_before)).sum() / target_probas_before.shape[0])
    confidence_increase = (target_probas_after_sharpening > target_probas_before).sum() / target_probas_before.shape[0]

    explanation_results = {
        'valid_samples_number': valid_samples.sum(),
        'valid_samples_ratio':  valid_samples.sum() / n_samples_original,
        'average_drop': average_drop,
        'confidence_increase': confidence_increase,
    }

    curves = dict(morf=dict(), lerf=dict())
    with job.stores['attribution_scores'] as attribution_scores:
        for order in ('morf', 'lerf'):
            scores_per_occlusion = [np.array(attribution_scores[f'{order}_8x8/{occlusion_idx}/scores']) for occlusion_idx in range(64)]
            target_scores_per_occlusion = [s[np.arange(n_samples_original), targets][valid_samples] for s in scores_per_occlusion]
            target_scores_per_occlusion = [target_scores_before] + target_scores_per_occlusion

            curves[order]['score'] = [s.mean() for s in target_scores_per_occlusion]

        for metric_type in ('multinomial', 'ordinal'):
            for order in ('morf', 'lerf'):
                for metric_name, m in metrics[metric_type]['labels'].items():
                    curves[order][metric_name] = [m(
                        np.array(attribution_scores['targets']),
                        np.array(attribution_scores[f'{order}_8x8/{occlusion_idx}/probas']).argmax(axis=1)
                    ) for occlusion_idx in range(64)]
                for metric_name, m in metrics[metric_type]['scores'].items():
                    curves[order][metric_name] = [m(
                        np.array(attribution_scores['targets']),
                        np.array(attribution_scores[f'{order}_8x8/{occlusion_idx}/probas'])
                    ) for occlusion_idx in range(64)]

    degradations = dict()
    for metric in curves['morf']:
        degradations[metric] = degradation(curves['morf'][metric], curves['lerf'][metric])

    explanation_results['curves'] = curves
    explanation_results['degradations'] = degradations
    
    job.doc['explanation_results'] = explanation_results


def increasingly_occluded_datamodules(
    original_datamodule: ExperimentDataModule,
    visual_explanations: Tensor,
    tiling: Tuple[int, int] = (8, 8),
    reverse: bool=False
) -> Iterator[pl.LightningDataModule]:

    tile_height, tile_width = tiling
    kernel_height, kernel_width = visual_explanations.size(2) // tile_height, visual_explanations.size(3) // tile_width
    kernel = torch.full((1, 1, kernel_height, kernel_width), 1 / (kernel_height * kernel_width)).float().cuda()
    tiled_explanations = F.conv2d(visual_explanations.cuda(), kernel, stride=(kernel_height, kernel_width))

    flattened_explanations = tiled_explanations.flatten(start_dim=1)
    orders = flattened_explanations.argsort(dim=1)

    def flat_idx2pixel_idx(flat_idx: int) -> Tuple[int, int]:
        return (flat_idx // tile_width, flat_idx % tile_height)

    masks = torch.ones((visual_explanations.size(0), 1, tile_height, tile_width)).float().cuda()
    for j in range(orders.size(1)):
        if reverse:
            pixel_idxs = [flat_idx2pixel_idx(orders[i, j].item()) for i in range(orders.size(0))]
        else:
            pixel_idxs = [flat_idx2pixel_idx(orders[i, -(j+1)].item()) for i in range(orders.size(0))]
        pixel_rows, pixel_cols = list(zip(*pixel_idxs))
        masks[np.arange(orders.size(0)), :, pixel_rows, pixel_cols] = 0.0
        resized_masks = resize(masks, list(original_datamodule.image_size), InterpolationMode.NEAREST).cpu()
        
        yield original_datamodule.masked_datamodule(resized_masks)


def degradation(morf_curve: List[float], lerf_curve: List[float]) -> float:
    first = morf_curve[0]
    last = morf_curve[-1]
    np_morf_curve = np.array(morf_curve)
    np_lerf_curve = np.array(lerf_curve)
    normalized_morf_curve = (np_morf_curve - last) / (first - last)
    normalized_lerf_curve = (np_lerf_curve - last) / (first - last)
    xs = np.linspace(0, 1, len(morf_curve))
    return scipy.integrate.trapezoid(normalized_lerf_curve, xs) - scipy.integrate.trapezoid(normalized_morf_curve, xs)


def seed_everything(op_name: str, *jobs: Job):
    bound_seed = hash(model_hash(jobs[0].sp))
    pl.seed_everything(bound_seed, workers=True)


def track_time(moment: str, op_name: str, *jobs: Job):
    import time

    current_time = time.strftime("%b %d, %Y at %l:%M:%S %p %Z")
    doc_key = f"{op_name}_{moment}_times"
    for job in jobs:
        job.doc.setdefault(doc_key, [])
        job.doc[doc_key].append(current_time)


if __name__ == '__main__':
    project = Project()
    project.project_hooks.on_start = [partial(track_time, 'start'), seed_everything]
    project.project_hooks.on_exit = [partial(track_time, 'exit')]
    project.main()
