import signac
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm


def add_job(project, sp, doc):
    if project.find_jobs(sp):
        return 0
    job = project.open_job(sp).init()
    for k, v in doc.items():
        job.doc[k] = v
    return 1


def main():
    project = signac.get_project()

    param_grid = {
        'seed': [0],
        'dataset_csv_path': ['data/smear/30holdout_80_10_10.csv', 'data/retinopathy/30holdout_80_10_10.csv', 'data/adience/30holdout_80_10_10.csv'],
        'partition': [0],
        'base_model': ['torchvision.models.resnet18'],
        'classifier_type': ['nominal'],
        'learning_rate': [1e-4],
        'patience': [20],
        'max_epochs': [5],
        'batch_size': [32, 64, 96, 128],
        'n_workers': [2, 3, 4, 5],
    }

    total_added = 0
    for p in tqdm(ParameterGrid(param_grid)):
        if 'smear' in p['dataset_csv_path']:
            p['smear_class_mapping'] = 'tbs'

        if 'resnet18' in p['base_model']:
            p['base_model_weights'] = 'torchvision.models.ResNet18_Weights.IMAGENET1K_V1'
        else:
            raise RuntimeError

        total_added += add_job(project, p, dict())
    print(f'Added a total of {total_added} new jobs')


if __name__ == '__main__':
    main()