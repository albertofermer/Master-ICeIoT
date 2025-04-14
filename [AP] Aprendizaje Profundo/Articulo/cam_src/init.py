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
        'dataset_csv_path': [
            'data/smear/30holdout_80_10_10.csv',
            'data/retinopathy/30holdout_80_10_10_full.csv',
            'data/adience/30holdout_80_10_10.csv',
            'data/cbis-ddsm/30holdout_80_10_10.csv',
        ],
        'partition': list(range(60)),
        'base_model': ['torchvision.models.resnet34'],
        'classifier_type': ['nominal', 'ordinal_ecoc', 'ordinal_unimodal_binomial_ce'],
        # 'explanation_method': ['gradcam', 'iba', 'iba_ce', 'gradcam++', 'scorecam', 'ablationcam', 'fullgrad', 'deeplift', 'ordinal_gradcam_binomial', 'ordinal_gradcam_obd_posneg'],
        'explanation_method': ['iba_ce'],
        #'step_exponent': [1, 2],
        'cam_layer': ['model.model[0].layer3[1].conv2'],
        'learning_rate': [1e-4],
        'patience': [20],
        'max_epochs': [200],
        'batch_size': [64],
        'n_workers': [2],
    }

    total_added = 0
    for p in tqdm(ParameterGrid(param_grid)):
        if (p['explanation_method'] == 'ordinal_gradcam_obd_posneg') and (p['classifier_type'] != 'ordinal_ecoc'):
            continue

        if 'smear' in p['dataset_csv_path']:
            p['smear_class_mapping'] = 'tbs'

        if 'csawm' in p['dataset_csv_path']:
            p['learning_rate'] = 1e-6
            p['batch_size'] = 32

        if 'resnet18' in p['base_model']:
            p['base_model_weights'] = 'torchvision.models.ResNet18_Weights.IMAGENET1K_V1'
        elif 'resnet34' in p['base_model']:
            p['base_model_weights'] = 'torchvision.models.ResNet34_Weights.IMAGENET1K_V1'
        else:
            raise RuntimeError

        total_added += add_job(project, p, dict())
    print(f'Added a total of {total_added} new jobs')


if __name__ == '__main__':
    main()