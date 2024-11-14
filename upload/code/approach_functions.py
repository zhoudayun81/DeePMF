#%%
import numpy as np
import pandas as pd
import pm4py, ast

RANDOM_SEED = 111
CASE_ID_KEY = 'Case ID'
ACTIVITY_KEY = 'Activity'
TIMESTAMP_KEY = 'Complete Timestamp'
#%%
def readcsv(file):
    # Read the CSV file with only three mandatory columns to fast process, parse timestamp during reading
    usecolumn = [CASE_ID_KEY, ACTIVITY_KEY, TIMESTAMP_KEY]
    event_log = pd.read_csv(file, usecols=usecolumn, dtype={CASE_ID_KEY: str, ACTIVITY_KEY: str}, parse_dates=[TIMESTAMP_KEY], na_filter=False, engine='c')
    return event_log

#%%
def build_directly_follows_matrix(activity_to_index, df_freqs, start_freqs, end_freqs):
    '''
    The df_matrix returned is interpreted as:
    From row activity To column activity.
    The last row is "Start", the last column is "End".
    E.g.: 
    activities = ['a', 'b', 'c']
    df_freqs = {('a', 'b'): 2, ('b', 'c'): 3}
    start_freqs = {'a': 2, 'b': 1}
    end_freqs = {'c': 3}
    df_matrix:
        a   b   c   E
    a   0   2   0   0
    b   0   0   3   0
    c   0   0   0   3   
    S   2   1   0   0 
    While the a, b, c, S, E won't show in the matrix, it's shown here for assisting understanding.
    Real output:
    array([[0, 2, 0, 0],
       [0, 0, 3, 0],
       [0, 0, 0, 3],
       [2, 1, 0, 0]])
    '''
    # Total number of unique activities
    N = len(activity_to_index)
    # Initialize the matrix with zeros
    df_matrix = np.zeros((N+1, N+1), dtype=int)
    # Fill in the directly-follows relationships
    for (start_activity, end_activity), freq in df_freqs.items():
        start_idx = activity_to_index[start_activity]
        end_idx = activity_to_index[end_activity]
        df_matrix[start_idx, end_idx] = freq
    # Fill in the start activities frequencies (last row)
    for activity, freq in start_freqs.items():
        idx = activity_to_index[activity]
        df_matrix[N, idx] = freq
    # Fill in the end activities frequencies (last column)
    for activity, freq in end_freqs.items():
        idx = activity_to_index[activity]
        df_matrix[idx, N] = freq
    return df_matrix

def reverse_dfg_from_dfm(df_matrix, activity_to_index):
    '''
    This function reverse the process of build_directly_follows_matrix in the format to be able to construct the dfg by pm4py.
    '''
    # Reverse the activity_to_index to get index_to_activity
    index_to_activity = {index: activity for activity, index in activity_to_index.items()}
    # Extract the list of activities
    activities = [index_to_activity[i] for i in range(len(activity_to_index))]
    # Initialize dictionaries for df_freqs, start_freqs, and end_freqs
    df_freqs = {}
    start_freqs = {}
    end_freqs = {}
    N = len(activities)
    # Reconstruct df_freqs from the matrix (ignoring the last row and column)
    for i in range(N):
        for j in range(N):
            if df_matrix[i, j] > 0:
                df_freqs[(index_to_activity[i], index_to_activity[j])] = df_matrix[i, j]
    # Reconstruct start_freqs from the last row (index N)
    for i in range(N):
        if df_matrix[N, i] > 0:
            start_freqs[index_to_activity[i]] = df_matrix[N, i]
    # Reconstruct end_freqs from the last column (index N)
    for i in range(N):
        if df_matrix[i, N] > 0:
            end_freqs[index_to_activity[i]] = df_matrix[i, N]
    '''Uncomment the following lines to show and save the dfg'''
    #pm4py.view_dfg(df_freqs, start_freqs, end_freqs, format='svg')
    #pm4py.save_vis_dfg(df_freqs, start_freqs, end_freqs, 'dfg.png')
    return activities, df_freqs, start_freqs, end_freqs


def get_directly_follows_matrices(event_log, num_window):
    unique_activities = event_log[ACTIVITY_KEY].unique()
    # Create a mapping from activity to its index in the matrix
    activity_to_index = {activity: i for i, activity in enumerate(unique_activities)}
    start_time = event_log[TIMESTAMP_KEY].min()
    end_time = event_log[TIMESTAMP_KEY].max()
    total_time_span = end_time - start_time
    window_size = total_time_span / num_window
    matrices = []
    for _ in range(num_window):
        end_time = start_time + window_size
        window_log = event_log[(event_log[TIMESTAMP_KEY] >= start_time) & (event_log[TIMESTAMP_KEY] <= end_time)]
        dfg, start_activities, end_activities = pm4py.discover_dfg(window_log, case_id_key=CASE_ID_KEY, activity_key=ACTIVITY_KEY, timestamp_key=TIMESTAMP_KEY)
        matrix = build_directly_follows_matrix(activity_to_index, dfg, start_activities, end_activities)
        matrices.append(matrix)
        start_time = end_time
    return matrices, activity_to_index

def remove_events_with_case_ids_in_range(df, start_time, end_time):
    # Filter the DataFrame based on the specified time range
    filtered_df = df[(df[TIMESTAMP_KEY] >= start_time) & (df[TIMESTAMP_KEY] <= end_time)]
    # Get the unique Case IDs
    ids_to_remove = filtered_df[CASE_ID_KEY].unique()
    return df[~df['Case ID'].isin(ids_to_remove)]

def sufficient_data(matrices, individual_matrix_sparsity_threshold=0.98, total_sparsity_threshold=0.2):
    sufficient = False
    total = len(matrices)
    count = 0
    for matrix in matrices:
        sparsity = np.sum(matrix == 0) / matrix.size
        value_sparsity = np.sum(matrix) / matrix.size
        if sparsity > individual_matrix_sparsity_threshold and value_sparsity < individual_matrix_sparsity_threshold:
            count+=1
    total_sparsity = count / total
    #print(f'total spar: {total_sparsity} count: {count}. Total: {total}')
    if total_sparsity < total_sparsity_threshold:
        sufficient = True
    return sufficient, total_sparsity

#%%
def tostring(my_list):
    # Convert list elements to strings
    string_list = []
    for item in my_list:
        if isinstance(item, dict):
            string_list.append(str(item))  # Convert dictionary to string representation
        else:
            string_list.append(item)  # Keep strings as they are

    # Join elements into a single string with a separator
    result_string = ', '.join(str(element) for element in string_list)
    return result_string

def saveDFMs(filename, matrices, activity_to_index):
    # Save the data to a .npz file
    np.savez_compressed(f'{filename}.npz', matrices=matrices, activity_to_index=activity_to_index)

def loadDFMs(filename):
    # Function to load data from .npz file
    # Load data from the .npz file
    data = np.load(f'{filename}', allow_pickle=True)
    # Extract matrices and text info
    matrices = data['matrices']
    activity_to_index = data['activity_to_index'].item()
    # Convert the arrays back to lists if necessary
    matrices = list(matrices)
    return matrices, activity_to_index

def loadPredDFM(filename):
    # Function to load data from .npz file
    # Load data from the .npz file
    data = np.load(f'{filename}', allow_pickle=True)
    # Extract matrices and text info
    matrices = data['matrices']
    activity_to_index = data['activity_to_index'].item()
    start_index = activity_to_index.find('{')
    cleaned_text = activity_to_index[start_index:]
    dict_obj = ast.literal_eval(cleaned_text)
    return matrices, dict_obj[0]

#%%
# the method uses all above methods for measurements
def measures(ground_truth_mat, forecast_mat):
    maev = float('inf')
    rmsdv = float('inf')
    mapev = float('inf')
    mse = float('inf')
    if forecast_mat.shape == ground_truth_mat.shape:
        rmsdv = rmsd(ground_truth_mat, forecast_mat)
        maev = mae(ground_truth_mat, forecast_mat)
        mapev = mape(ground_truth_mat, forecast_mat)
        mse = np.mean((forecast_mat - ground_truth_mat)**2)
    return {'rmsd':rmsdv, 'mae':maev, 'mape':mapev, 'mse':mse}

def rmsd(ground_truth_mat, forecast_mat):
    rmsd = float('inf')
    # doesn't measure rmsd if the forecast is broken (i.e. less than expected)
    if forecast_mat.shape == ground_truth_mat.shape:
        # Root-mean-square deviation
        diff = forecast_mat - ground_truth_mat
        squared_diff = np.square(diff)
        mean_squared_diff = np.mean(squared_diff)
        rmsd = np.sqrt(mean_squared_diff)
    return rmsd

def mae(ground_truth_mat, forecast_mat):
    mae = float('inf')
    if forecast_mat.shape == ground_truth_mat.shape:
        # Mean Average Error
        diff = forecast_mat - ground_truth_mat
        abs_diff = np.abs(diff)
        mae = np.mean(abs_diff)
    return mae 

def mape(ground_truth_mat, forecast_mat):
    mape = float('inf')
    if forecast_mat.shape == ground_truth_mat.shape:
        # Calculate absolute percentage error element-wise
        abs_percentage_error = np.abs((ground_truth_mat - forecast_mat) / ground_truth_mat)
        # Handle cases where y_true is zero to avoid division by zero errors
        abs_percentage_error[np.isnan(abs_percentage_error)] = 0.0
        abs_percentage_error[np.isinf(abs_percentage_error)] = 0.0
        # Calculate MAPE
        mape = np.mean(abs_percentage_error) * 100
    return mape
