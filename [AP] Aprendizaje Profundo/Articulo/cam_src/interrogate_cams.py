#!/home/javierbg/miniconda3/envs/cam/bin/python

from project import load_data, build_model_checkpoint, last_checkpoint_path
from model import ExperimentModel, AttributionModel
import signac
from signac.contrib.job import Job
from torchvision.models.quantization.utils import _replace_relu as replace_relu
import pytorch_lightning as pl
import torch
from data import InvertNormalization
import warnings
import numpy as np
from PIL import Image
from data import ExperimentDataModule
import time
import click
from pathlib import Path


warnings.filterwarnings("ignore", message=r"Selected estimator was only fitted on \d+ samples", category=UserWarning, module='IBA')
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
inverse_normalization = InvertNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

@click.command()
@click.argument('device')
@click.argument('n_devices', type=int)
@click.argument('method')
def main(device: str, n_devices: int, method: str):
    pr = signac.get_project()

    dataset_csv_paths = [
        'data/smear/30holdout_80_10_10.csv',
        'data/retinopathy/30holdout_80_10_10_full.csv',
        'data/adience/30holdout_80_10_10.csv',
        'data/cbis-ddsm/30holdout_80_10_10.csv',
    ]
    classifier_types = ['nominal', 'ordinal_ecoc', 'ordinal_unimodal_binomial_ce']
    explanation_methods = ['gradcam', 'iba']
    n = 10

    DATAMODULE_CACHE: dict[str, ExperimentDataModule] = dict()

    def get_datamodule(job: Job) -> ExperimentDataModule:
        if job.sp.dataset_csv_path in DATAMODULE_CACHE:
            return DATAMODULE_CACHE[job.sp.dataset_csv_path]

        datamodule = load_data(job)
        datamodule.setup('predict', n)
        DATAMODULE_CACHE[job.sp.dataset_csv_path] = datamodule
        return datamodule


    # for dataset_csv_path, classifier_type, explanation_method in tqdm(product(dataset_csv_paths, classifier_types, explanation_methods), total=24):
    #     dataset_name = Path(dataset_csv_path).parts[1]
    #     job = list(pr.find_jobs({
    #         'classifier_type': classifier_type,
    #         'dataset_csv_path': dataset_csv_path,
    #         'explanation_method': explanation_method
    #     }))[0]

    for job in pr.find_jobs({'dataset_csv_path': 'data/smear/30holdout_80_10_10.csv', 'classifier_type': 'ordinal_unimodal_binomial_ce', 'explanation_method': method, 'partition': 0}):
        checkpoint_callback = build_model_checkpoint(job)
        datamodule = get_datamodule(job)

        assert (lcp := last_checkpoint_path(checkpoint_callback)) is not None
        model = ExperimentModel.load_from_checkpoint(str(lcp), n_classes=datamodule.n_classes, job=job)
        replace_relu(model)

        trainer = pl.Trainer(deterministic='warn', accelerator=device, devices=n_devices, enable_progress_bar=False,
                                inference_mode=False, logger=False)
        
        # mod_sp = dict(job.sp)
        # mod_sp['explanation_method'] = 'fullgrad'
        # mod_job = pr.open_job(mod_sp)
        # mod_job.doc = job.doc
        mod_job = job

        start_time = time.perf_counter()
        attribution_model = AttributionModel(mod_job, model, datamodule)
        explanations = torch.cat(trainer.predict(attribution_model, datamodule)).detach().cpu().numpy()  # type: ignore
        elapsed_time = time.perf_counter() - start_time
        print(f'Elapsed time: {elapsed_time} seconds')

        method_output_path = Path(f'samples/{method}')
        method_output_path.mkdir(exist_ok=True)
        for i in range(10):
            Image.fromarray((explanations[i, 0]*255).astype(np.uint8), 'L').save(str(method_output_path / f'{method}_{device}_mod_{i}.png'))
            # Image.fromarray((explanations[i].transpose((1,2,0))*255).astype(np.uint8), 'RGB').save(str(method_output_path / f'{method}_{device}_mod_{i}.png'))
        #Image.fromarray((explanations[9, 0]*255).astype(np.uint8), 'L').save(f'pru_scorecam_mod_9.png')

        # for i in range(len(datamodule.predict_dataset)):
        #     datamodule.predict_dataset.untransformed_item(i)[0].save(f'example_explanations/{dataset_name}_{classifier_type}_{explanation_method}_{i}_input.png')
        #     Image.fromarray((explanations[i,0]*255).astype(np.uint8), 'L').save(f'example_explanations/{dataset_name}_{classifier_type}_{explanation_method}_{i}_expl.png')
        break


if __name__ == '__main__':
    main()