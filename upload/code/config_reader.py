import  os, sys, logging, configparser

config_file = 'config.ini'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE =  os.path.join(CURRENT_DIR, config_file)

if not os.path.exists(CONFIG_FILE): 
    logging.error("Config file does not exist! Program exit. Please check the file is located and named at: %s"%CONFIG_FILE)
    sys.exit()

CONFIG = configparser.ConfigParser(allow_no_value=True)
CONFIG.read(CONFIG_FILE)
SECTIONS = CONFIG.sections()

def process_values(value, dtype):
    values = value.split()
    if dtype == 'categorical':
        return values
    elif dtype == 'int':
        int_values = sorted(map(int, values))
        if len(int_values) > 2:
            logging.error("Wrong config for epochs! Need to set only two values indicating a range. Use the smallest and largest value as the range bound.")
            return [int_values[0], int_values[-1]]
        elif len(int_values) <= 1:
            logging.error("Wrong config for epochs! Need to set two values indicating a range. Use the only value as the upper bound from 0.")
            return [0, int_values[0]]
        else:
            return int_values
    elif dtype == 'float':
        float_values = sorted(map(float, values))
        if len(float_values) > 2:
            logging.error("Wrong config for epochs! Need to set only two values indicating a range. Use the smallest and largest value as the range bound.")
            return [float_values[0], float_values[-1]]
        elif len(float_values) <= 1:
            logging.error("Wrong config for epochs! Need to set two values indicating a range. Use the only value as the upper bound from 0.")
            return [0.0, float_values[0]]
        else:
            return float_values
        
# --------- Universal Program Parameters ---------
INPUT_DIR = CONFIG['Program']['input_dir']
OUTPUT_DIR = CONFIG['Program']['output_dir']
ANA_DIR = CONFIG['Program']['analysis_dir']
LOG_DIR = CONFIG['Program']['log_dir']
MODEL_DIR = CONFIG['Program']['model_dir']
# --------- End of Universal Program Parameters ---------

# --------- Select experiment models ---------
MODEL = process_values(CONFIG.get('Model','model'), 'categorical')
# --------- End of Select experiment models ---------

# --------- Hyperparameter Space ---------
OPTIMIZER = process_values(CONFIG.get('Hyperparameters','optimizer'), 'categorical')
LOSS_FUNCTION = process_values(CONFIG.get('Hyperparameters','loss_function'), 'categorical')
hidden_size = process_values(CONFIG.get('Hyperparameters','hidden_size'), 'categorical')
HIDDEN_SIZE = [int(x) for x in hidden_size]

EPOCHS = process_values(CONFIG.get('Hyperparameters','epochs'), 'int')
#NUM_LAYERS = process_values(CONFIG.get('Hyperparameters','num_layers'), 'int')
NUM_LAYERS = CONFIG['Hyperparameters'].getint('num_layers')
#BATCH_SIZE = process_values(CONFIG.get('Hyperparameters','batch_size'), 'int')
BATCH_SIZE = CONFIG['Hyperparameters'].getint('batch_size')
#KERNEL_SIZE = process_values(CONFIG.get('Hyperparameters','kernel_size'), 'int')
KERNEL_SIZE = CONFIG['Hyperparameters'].getint('kernel_size')

LEARNING_RATE = process_values(CONFIG.get('Hyperparameters','learning_rate'), 'float')
DROPOUT_PROBABILITY = process_values(CONFIG.get('Hyperparameters','dropout_probability'), 'float')
# --------- End of Hyperparameter Space ---------

# --------- Approach Specific Parameters ---------
NFOLD = CONFIG['Approach-Specific'].getint('nfold')
TRIAL = CONFIG['Approach-Specific'].getint('trial')
WINDOW = CONFIG['Approach-Specific'].getint('window') if CONFIG['Approach-Specific'].get('window') is None else CONFIG['Approach-Specific']['window']
EVAL = CONFIG['Approach-Specific'].getboolean('eval')
WEIGHT = process_values(CONFIG.get('Approach-Specific','condition_weight'), 'categorical')
# --------- End of Approach Specific Parameters ---------