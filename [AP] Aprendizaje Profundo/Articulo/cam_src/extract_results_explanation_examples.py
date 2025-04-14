from pathlib import Path
import signac
import click
import project
from model import ExperimentModel, AttributionModel
from typing import cast
from torchvision.models.quantization.utils import _replace_relu as replace_relu  # type: ignore
import pytorch_lightning as pl
import torch
from PIL import Image
import numpy as np


datasets_indexes = {
    'data/smear/30holdout_80_10_10.csv': dict(name='Herlev', partition=0, test_idxs=list(range(20))),
    'data/retinopathy/30holdout_80_10_10_full.csv': dict(name='Retinopathy', partition=0, test_idxs=list(range(20))),
    'data/adience/30holdout_80_10_10.csv': dict(name='Adience', partition=0, test_idxs=list(range(20))),
    'data/cbis-ddsm/30holdout_80_10_10.csv': dict(name='CBIS-DDSM', partition=0, test_idxs=list(range(20))),
}

explanation_methods = {
    'gradcam_score': 'Grad-CAM',
    'gradcam++_score': 'Grad-CAM++',
    'scorecam': 'Score-CAM',
    'ordinal_gradcam_obd_posneg': 'GradOBD-CAM',
    'iba_ce': 'IBA',
    'iba': 'OIBA',
}


@click.command()
@click.argument('output_folder_path', type=click.Path(file_okay=False, path_type=Path))
def main(output_folder_path: Path):
    if not output_folder_path.exists():
        output_folder_path.mkdir(parents=True)
    
    pr = signac.get_project()

    for dataset_csv_path, dataset_params in datasets_indexes.items():
        for explanation_method, em_name in explanation_methods.items():
            jobs = list(pr.find_jobs({
                'dataset_csv_path': dataset_csv_path,
                'explanation_method': explanation_method,
                'classifier_type': 'ordinal_ecoc',
                'partition': dataset_params['partition'],
            }))
            assert len(jobs) == 1
            job = jobs[0]
            datamodule = project.load_data(job)
            dataset = datamodule.non_modified_dataset()
            for test_idx in dataset_params['test_idxs']:
                p = output_folder_path / f'{dataset_params["name"]}_input_{test_idx}.png'
                if p.exists():
                    continue
                if test_idx >= len(dataset):
                    continue

                input_sample, _ = dataset[test_idx]
                input_sample_img = Image.fromarray((input_sample.numpy()*255).astype(np.uint8).transpose((1,2,0)), 'RGB')
                input_sample_img.save(p)

            checkpoint_callback = project.build_model_checkpoint(job)

            assert (lcp := project.last_checkpoint_path(checkpoint_callback)) is not None
            model = cast(ExperimentModel, ExperimentModel.load_from_checkpoint(str(lcp), n_classes=datamodule.n_classes, job=job))
            replace_relu(model)

            trainer = pl.Trainer(deterministic='warn', accelerator='gpu', devices=1, enable_progress_bar=False,
                                inference_mode=False)
            model = model.to('cuda:0')

            attribution_model = AttributionModel(job, model, datamodule)
            explanations = torch.cat(trainer.predict(attribution_model, datamodule))  # type: ignore

            for test_idx in dataset_params['test_idxs']:
                if test_idx >= len(dataset):
                    continue

                explanation = explanations[test_idx, 0]
                explanation_img = Image.fromarray((explanation.numpy() * 255).astype(np.uint8), 'L')
                explanation_img.save(output_folder_path / f'{dataset_params["name"]}_{em_name}_explanation_{test_idx}.png')
    


if __name__ == '__main__':
    main()