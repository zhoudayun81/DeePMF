if __name__ == '__main__':
    import os, datetime, logging, multiprocessing, sys, torch
    # import my own modules
    import config_reader, worker

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
    THREADS = 6

    # Create a logger for the main process
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(LOG_DIR, datetime.datetime.now().strftime('%Y-%m-%d_%H=%M=%S') + '_main.log')
    fh = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    for section in config_reader.SECTIONS:
        for key in config_reader.CONFIG[section]:
            logger.info('Read from section %s, key %s, and its value = %s.' % (section, key, config_reader.CONFIG[section][key]))
    logger.info('CURRENT_FOLDER=%s'%CURRENT_FOLDER)
    logger.info('PROJECT_DIR=%s'%PROJECT_DIR)
    logger.info('DOWNLOAD_DIR=%s'%DOWNLOAD_DIR)
    logger.info('UPLOAD_DIR=%s'%UPLOAD_DIR)
    logger.info('INPUT_DIR=%s'%INPUT_DIR)
    logger.info('OUTPUT_DIR=%s'%OUTPUT_DIR)
    logger.info('LOG_DIR=%s'%LOG_DIR)
    logger.info('MODEL_DIR=%s'%MODEL_DIR)
    logger.info('Total %s number of files to experiment. INPUTFILES=%s'%(len(INPUTFILES), INPUTFILES))

    if INPUTFILES and len(config_reader.MODEL)>1 and config_reader.OPTIMIZER and config_reader.LEARNING_RATE and config_reader.LOSS_FUNCTION:
        torch.multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool() as pool:
            pool.starmap(worker.model_worker, [(INPUTFILES, INPUT_DIR, OUTPUT_DIR, LOG_DIR, MODEL_DIR, model) for model in config_reader.MODEL])
    elif INPUTFILES and config_reader.MODEL and config_reader.OPTIMIZER and config_reader.LEARNING_RATE and config_reader.LOSS_FUNCTION:
        file_groups = [INPUTFILES[i::THREADS] for i in range(THREADS)]
        file_groups = [group for group in file_groups if len(group) > 0]
        torch.multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool() as pool:
            pool.starmap(worker.model_worker, [(files, INPUT_DIR, OUTPUT_DIR, LOG_DIR, MODEL_DIR, config_reader.MODEL[0]) for files in file_groups])
    else:
        logger.error('Required parameter(s) or input files missing! Check the configuration file! Program exit!')
        sys.exit()