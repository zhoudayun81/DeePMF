#%%
import os, logging, multiprocessing, datetime, optuna
from optuna.trial import TrialState
from optuna.exceptions import OptunaError
# import my own modules
import config_reader
import models as m
import approach_functions as af

def model_worker(inputfiles, inputdir, outputdir, log_path, model_dir, model_name):
    # Create a logger for this worker process
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(log_path, datetime.datetime.now().strftime('%Y-%m-%d_%H=%M=%S') + f'_{multiprocessing.current_process().name}_{model_name}.log')
    fh = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f"{model_name} selected for the thread.")
    for filename in inputfiles:
        # ----------------- Data Preprocessing -----------------
        matrice_file_path = os.path.join(inputdir, filename)
        if not os.path.isfile(matrice_file_path):
            logger.info(f'{filename} does not exits in path {inputdir}, skipping this file.')
            continue
        matrices, activity_info = af.loadDFMs(matrice_file_path)
        individual_pair = m.One2OneDataset(matrices)
        logger.info(f"{filename} total {len(matrices)} number of matrices, and {len(individual_pair)} pairs.")
        train_loaders, val_loaders, test_loaders = m.get_data_loaders(individual_pair, config_reader.NFOLD, config_reader.BATCH_SIZE)
        # ----------------- End of Data Preprocessing -----------------
        # ----------------- Training -----------------
        for fold in range(config_reader.NFOLD):
            train_loader = train_loaders[fold]
            val_loader = val_loaders[fold]
            test_loader = test_loaders[fold]
            logger.info(f'Fold {fold+1}/{config_reader.NFOLD}')

            study_name=f'{filename[:-4]}_fold{fold+1}'
            db_file = f'sqlite:///{os.path.join(log_path, model_name)}.db'
            try:
                study = optuna.load_study(study_name=study_name, storage=db_file)
                logger.info(f"{study_name} loaded successfully.")
            except (OptunaError, KeyError) as e:
                study = optuna.create_study(study_name=study_name, storage=db_file, direction='minimize', load_if_exists=True)
                logger.info(f"{db_file} {study_name} created.")
            completed_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE or t.state == TrialState.PRUNED])
            logger.info(f"Number of completed trials: {completed_trials} in study")
            left_trials = config_reader.TRIAL - completed_trials
            if left_trials > 0:
                logger.info(f"Running {left_trials} number of trials")
                study.optimize(lambda trial: m.objective(trial, model_name, activity_info, train_loader, val_loader), n_trials=left_trials)
                logger.info(f"Best hyperparameters: {study.best_params}")
                logger.info(f"Best validation loss: {study.best_value}")
            # Tunning complete
            if config_reader.EVAL:
                # Save the best model
                universal_name = f'{filename[:-4]}_{model_name}_{fold+1}'
                pred = os.path.join(outputdir, f'{universal_name}.npz')
                if os.path.isfile(pred):
                    logger.info(f"{pred} exits, skip eval and continue to the next one.")
                    continue
                best_model_params = study.best_params
                best_model, criterion = m.train_best(best_model_params, model_name, activity_info, train_loader)
                state_dict_path = os.path.join(model_dir, universal_name + '_best_state_dict.pt')
                text_path = os.path.join(model_dir, universal_name + f'_hyperparams.txt')
                best_state_dict = best_model.state_dict()
                m.save_state_dict(best_state_dict, state_dict_path, best_model_params, text_path)
                logger.info(f"{model_name} Best model for fold {fold+1} saved!")
                # ----------------- End of Training -----------------
                # After training, test the training dataset accuracy using the best training params
                best_model.load_state_dict(best_state_dict)
                best_model.eval()
                # with torch.no_grad(): is called in test method
                # ----------------- Testing -----------------
                test_loss, prediction, ground_truths = m.evaluate_model(best_model, model_name, test_loader, criterion)
                measures = af.measures(ground_truths[0], prediction[0])
                predicted_matrix_path = os.path.join(outputdir,universal_name)
                mat_info = []
                mat_info.append(filename)
                mat_info.append(activity_info)
                #mat_info.append({'test loss':test_loss})
                mat_info.append(measures)
                info_string = af.tostring(mat_info)
                logger.info(f'Results: {info_string}')
                af.saveDFMs(predicted_matrix_path, prediction[0], info_string)
                logger.debug(f'{universal_name}. Prediction generated at:{predicted_matrix_path}')
#%% Test on no threads when run from this file
if __name__ == '__main__':
    upload_folder = 'upload'
    download_folder = 'download'
    input_folder = 'input'
    code_folder = 'code'
    output_folder = 'output'
    log_folder = 'log'
    model_folder = 'model'

    # The project working directory on Spartan should be: /data/gpfs/projects/punim1925/xxx
    CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_FOLDER))
    DOWNLOAD_DIR = os.path.join(PROJECT_DIR, download_folder)
    UPLOAD_DIR = os.path.join(PROJECT_DIR, upload_folder)
    # The code is by default that directory traversal doesn't apply.
    INPUT_DIR = config_reader.INPUT_DIR if config_reader.INPUT_DIR else os.path.join(UPLOAD_DIR, input_folder)
    OUTPUT_DIR = config_reader.OUTPUT_DIR if config_reader.OUTPUT_DIR else os.path.join(DOWNLOAD_DIR, output_folder)
    LOG_DIR = config_reader.LOG_DIR if config_reader.LOG_DIR else os.path.join(DOWNLOAD_DIR, log_folder)
    MODEL_DIR = config_reader.MODEL_DIR if config_reader.MODEL_DIR else os.path.join(DOWNLOAD_DIR, model_folder)
    INPUTFILES = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f)) and f.endswith(f'_{config_reader.WINDOW}.npz') and not f.startswith('.')] if config_reader.WINDOW else [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f)) and f.endswith('.npz') and not f.startswith('.')] # if window is given, then only pick the time window files to run, otherwise run all

    for model in config_reader.MODEL:
        model_worker(INPUTFILES, INPUT_DIR, OUTPUT_DIR, LOG_DIR, MODEL_DIR, model)
# %%
