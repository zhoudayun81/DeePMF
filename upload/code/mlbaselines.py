#%%
import numpy as np
import pandas as pd
import os, logging, datetime
from statsmodels.tsa.stattools import adfuller
#from sklearn.preprocessing import StandardScaler

import approach_functions as af

from statsmodels.tsa.api import VAR
def fit_var_model(train_data, forecast_horizon):
    # Fit the VAR model with lag 1
    model = VAR(train_data)
    model_fitted = model.fit(1)  # Fit the VAR(1) model
    # data.values[-model_fitted.k_ar:] extracts the last k_ar observations required for forecasting (where k_ar is the number of lags used in the model).
    forecast = model_fitted.forecast(train_data.values[-model_fitted.k_ar:], steps=forecast_horizon)
    return forecast

from statsmodels.tsa.statespace.varmax import VARMAX
def fit_varma_model(train_data, forecast_horizon):
    # Identify columns that contain only zeros
    zero_columns = train_data.columns[(train_data == 0).all()]
    # Record the column indexes (positions)
    zero_column_indexes = {col: train_data.columns.get_loc(col) for col in zero_columns}
    # Drop columns with only zeros
    train_data = train_data.drop(columns=zero_columns)
    model = VARMAX(train_data.astype(np.float64), order=(1, 1))
    model_fit = model.fit()
    forecast_values = model_fit.forecast(steps=forecast_horizon)
    processed_df = forecast_values  # Replace this with actual processing steps
    # Reinsert the zero columns at their original positions
    for col, idx in zero_column_indexes.items():
        processed_df.insert(idx, col, 0)
    return processed_df.round().astype(int)

from statsmodels.tsa.api import VARMAX
def fit_vma_model(train_data, forecast_horizon):
    # Identify columns that contain only zeros
    zero_columns = train_data.columns[(train_data == 0).all()]
    # Record the column indexes (positions)
    zero_column_indexes = {col: train_data.columns.get_loc(col) for col in zero_columns}
    # Drop columns with only zeros
    train_data = train_data.drop(columns=zero_columns)
    model = VARMAX(train_data.astype(np.float64), order=(0, 2))
    model_fit = model.fit()
    forecast_values = model_fit.forecast(steps=forecast_horizon)
    processed_df = forecast_values  # Replace this with actual processing steps
    # Reinsert the zero columns at their original positions
    for col, idx in zero_column_indexes.items():
        processed_df.insert(idx, col, 0)
    return processed_df

def identity_model(train_data, forecast_horizon):
    forecast = train_data[-forecast_horizon:]  # Naive forecast using the last observed value
    return forecast

#Naive Model (Average Forecast)
#The naive model forecasts future values based on the average of past observations.
def fit_naive_agerage_model(train_data):
    # Compute the mean of each column
    forecast = train_data.mean().round().astype(int).to_numpy().reshape(1, -1)
    return forecast

#univariate model
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def hw_forecast(series, steps=1):
    model = ExponentialSmoothing(
        series
    ).fit()
    forecast = model.forecast(steps)
    return forecast

from statsmodels.tsa.ar_model import AutoReg
def ar_forecast(series, steps=1, lags=1):
    model = AutoReg(series, lags=lags).fit()
    forecast = model.predict(start=len(series), end=len(series) + steps - 1)
    return forecast

from statsmodels.tsa.arima.model import ARIMA
def arima_forecast(series, order, steps=1):
    model = ARIMA(series, order=order).fit()
    forecast = model.forecast(steps)
    return forecast

def reshape_matrices_to_1d(matrices):
    reshaped = [matrix.flatten() for matrix in matrices]
    return reshaped

def reshape_1d_to_matrix(array):
    N = int(array.shape[1]**0.5)
    reshaped = array.reshape(N, N)
    return reshaped

def split_into_folds(data, n_chunks, validation_size=0, test_size=1):
    n = len(data)
    chunk_size = n // n_chunks
    folds = {}
    for i in range(1, n_chunks + 1):
        fold_data = data[:i * chunk_size + 1]
        train_size = len(fold_data) - validation_size - test_size 
        train_set = fold_data[:train_size]
        validation_set = fold_data[train_size:train_size + validation_size]
        test_set = fold_data[train_size + validation_size:]
        folds[f'fold_{i}'] = {
            'train': train_set,
            'validation': validation_set,
            'test': test_set
        }
    return folds

# Function to check stationarity for all columns
def check_stationarity(df, significance_level=0.05):
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    # Optionally, drop constant columns
    train_data_non_constant = train_data.drop(columns=constant_columns)
    non_stationary_columns = []
    for column in train_data_non_constant.columns:
        result = adfuller(train_data_non_constant[column])
        p_value = result[1]
        #print(f'Column: {column} | ADF Statistic: {result[0]:.4f} | p-value: {p_value:.4f}')
        if p_value > significance_level:
            print(f"--> {column} is likely non-stationary.\n")
            non_stationary_columns.append(column)
    return non_stationary_columns

#%%
if __name__ == '__main__':
    forecast_horizon = 1
    nfold = 10
    val_size = 0
    test_size = 1

    INPUT_DIR =''
    LOG_DIR =''
    OUTPUT_DIR = ''
    logger = logging.getLogger('baselines')
    log_name = os.path.join(LOG_DIR, datetime.datetime.now().strftime('%Y-%m-%d_%H=%M=%S') + '_baselines.log')
    logging.basicConfig(filename=log_name, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger.info(f'nfold: {nfold}, forecast_horizon: {forecast_horizon}, val_size: {val_size}, test_size: {test_size}')

    inputfiles = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f)) and f.endswith('.npz') and not f.startswith('.')] #list of file names with extension
    model_data = {}
    model_data['identityFunction'] = []
    model_data['naiveAverage'] = []
    model_data['var'] = []
    model_data['hw'] = []
    model_data['ar'] = []
    model_data['arima'] = []
    for filename in inputfiles:
        logger.info(f'Filename: {filename}')
        print(f'Filename: {filename}')
        matrice_file_path = os.path.join(INPUT_DIR, filename)
        matrices, activity_info = af.loadDFMs(matrice_file_path)
        last_index = filename.rfind('_')
        file_name = filename[:last_index]
        lag = int(filename[last_index + 1:-4])
        flat_matrices = reshape_matrices_to_1d(matrices)
        folds = split_into_folds(flat_matrices, nfold, val_size, test_size)
        for fold, datasets in folds.items():
            logger.info(f'Fold: {fold}')
            train_data = datasets['train']
            train_data = pd.DataFrame(train_data)
            train_data = train_data.astype(np.float64)
            test_data = reshape_1d_to_matrix(datasets['test'][0].reshape(1, -1))
            
            forecast_identity_function = identity_model(train_data, forecast_horizon)
            forecast_identity_function = reshape_1d_to_matrix(forecast_identity_function.to_numpy())
            af.saveDFMs(f'{OUTPUT_DIR}/{file_name}_{lag}_identityFunction_{fold[5:]}', forecast_identity_function, activity_info)
            identity_function_results = af.measures(test_data, forecast_identity_function)
            logger.info(f'identityFunction-results: {identity_function_results}')

            forecast_naive_average = fit_naive_agerage_model(train_data)
            forecast_naive_average = reshape_1d_to_matrix(forecast_naive_average)
            af.saveDFMs(f'{OUTPUT_DIR}/{file_name}_{lag}_naiveAverage_{fold[5:]}', forecast_naive_average, activity_info)
            naive_average_results = af.measures(test_data, forecast_naive_average)
            logger.info(f'naiveAverage-results: {naive_average_results}')
            
            try:
                forecast_var = fit_var_model(train_data, forecast_horizon)
                forecast_var = reshape_1d_to_matrix(forecast_var)
                af.saveDFMs(f'{OUTPUT_DIR}/{file_name}_{lag}_var_{fold[5:]}', forecast_var, activity_info)
                var_results = af.measures(test_data, forecast_var)
                logger.info(f'var-results: {var_results}')
            except ValueError as e:
                logger.error(f"ValueError: {e}")
            except RuntimeError as e:
                logger.error(f"RuntimeError: {e}")
            except Exception as e:
                logger.exception(f"An unexpected error occurred: {e}")

            hw_pred_mat = np.zeros(datasets['test'][0].reshape(1, -1).shape)
            ar_pred_mat = np.zeros(datasets['test'][0].reshape(1, -1).shape)
            arima_pred_mat = np.zeros(datasets['test'][0].reshape(1, -1).shape)
            for i in train_data.columns:
                hw_predictions = hw_forecast(train_data[i].values, steps=1)
                hw_pred_mat[:,i] = hw_predictions
                ar_predictions = ar_forecast(train_data[i].values, steps=1, lags=2)
                ar_pred_mat[:,i] = ar_predictions
                arima_predictions = arima_forecast(train_data[i].values, order=(1, 1, 1), steps=1)
                arima_pred_mat[:,i] = arima_predictions
            hw_pred_mat = reshape_1d_to_matrix(hw_pred_mat)
            af.saveDFMs(f'{OUTPUT_DIR}/{file_name}_{lag}_hw_{fold[5:]}', hw_pred_mat, activity_info)
            hw_function_results = af.measures(test_data, hw_pred_mat)
            logger.info(f'hw-results: {hw_function_results}')

            ar_pred_mat = reshape_1d_to_matrix(ar_pred_mat)
            af.saveDFMs(f'{OUTPUT_DIR}/{file_name}_{lag}_ar_{fold[5:]}', ar_pred_mat, activity_info)
            ar_function_results = af.measures(test_data, ar_pred_mat)
            logger.info(f'ar-results: {ar_function_results}')

            arima_pred_mat = reshape_1d_to_matrix(arima_pred_mat)
            af.saveDFMs(f'{OUTPUT_DIR}/{file_name}_{lag}_arima_{fold[5:]}', arima_pred_mat, activity_info)
            arima_function_results = af.measures(test_data, arima_pred_mat)
            logger.info(f'arima-results: {arima_function_results}')

            #check_stationarity(train_data)
            '''scaler = StandardScaler()
            train_data_scaled = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
            # the following multivariate baseline models do not work.
            try:
                forecast_varma = fit_varma_model(train_data_scaled, forecast_horizon)
                forecast_varma = reshape_1d_to_matrix(forecast_varma)
                varma_results = af.measures(test_data, forecast_varma)
                logger.info(f'varma_results: {varma_results}')
            except ValueError as e:
                logger.error(f"ValueError: {e}")
            except RuntimeError as e:
                logger.error(f"RuntimeError: {e}")
            except Exception as e:
                logger.exception(f"An unexpected error occurred: {e}")
            try:
                forecast_vma = fit_vma_model(train_data_scaled, forecast_horizon)
                forecast_vma = reshape_1d_to_matrix(forecast_vma)
                vma_results = af.measures(test_data, forecast_vma)
                logger.info(f'vma_results: {vma_results}')
            except ValueError as e:
                logger.error(f"ValueError: {e}")
            except RuntimeError as e:
                logger.error(f"RuntimeError: {e}")
            except Exception as e:
                logger.exception(f"An unexpected error occurred: {e}")'''
# %%
