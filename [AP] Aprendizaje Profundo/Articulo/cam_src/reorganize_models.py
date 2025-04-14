import signac
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional
from pathlib import Path
from signac.contrib.job import Job
import sys
import shutil

base_model_attributes = ['seed', 'dataset_csv_path', 'partition', 'base_model', 'classifier_type', 'learning_rate', 'patience', 'max_epochs', 'batch_size']

def build_model_checkpoint(job: Job) -> ModelCheckpoint:
    checkpoint_dirpath = job.fn('checkpoint')
    return ModelCheckpoint(checkpoint_dirpath, monitor='val_loss', mode='min', save_last=True)


def last_checkpoint_path(mc: ModelCheckpoint) -> Optional[Path]:
    last_checkpoint_path = Path(f'{mc.dirpath}/{mc.CHECKPOINT_NAME_LAST}{mc.FILE_EXTENSION}')
    if last_checkpoint_path.is_file():
        return last_checkpoint_path
    else:
        return None

def model_hash(statepoint) -> str:
    """
    Generates a hex token that identifies a model
    From: https://stackoverflow.com/a/69669240/4174961
    """
    config = tuple(sorted([(k, v) for k, v in statepoint.items() if k in base_model_attributes], key=lambda e: e[0]))
    # `sign_mask` is used to make `hash` return unsigned values
    sign_mask = (1 << sys.hash_info.width) - 1
    # Get the hash as a positive hex value with consistent padding without '0x'
    return f'{hash(config) & sign_mask:#0{sys.hash_info.width//4}x}'[2:]

pr = signac.get_project()
new_dir = Path('checkpoints')
new_dir.mkdir(exist_ok=True)
for attrs, js in pr.groupby(base_model_attributes):
    job_list = list(js)
    if not any(Path(j.fn('checkpoint')).exists() for j in job_list):
        continue

    real_jobs = [j for j in job_list if (Path(j.fn('checkpoint')).exists()) and (not Path(j.fn('checkpoint')).is_symlink())]
    if not real_jobs:
        continue
    real_job = real_jobs[0]
    real_job_checkpoint = Path(real_job.fn('checkpoint'))
    assert not real_job_checkpoint.is_symlink()
    h = model_hash(real_job.sp)
    real_job_checkpoint.rename(new_dir / h)
    for j in job_list:
        checkpoint_dir = Path(j.fn('checkpoint'))
        if checkpoint_dir.exists():
            if checkpoint_dir.is_symlink():
                checkpoint_dir.unlink()
            elif checkpoint_dir.is_dir():
                shutil.rmtree(str(checkpoint_dir))
            else:
                print(f'WARNING: job {j.id}')
    