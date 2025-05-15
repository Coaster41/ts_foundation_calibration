import numpy as np
import pandas as pd


def mae(results_df, data_df, freq_delta):
    pred_length = int(results_df.columns[-1])
    mae_arr = []
    for h in range(1,pred_length+1):
        shift_results = results_df[['ds', 'unique_id', str(h)]]
        shift_results.loc[:,'ds'] += freq_delta * h
        merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
        mean_abs_error = np.mean(np.abs(merged_results['y'] - merged_results[str(h)]))
        mae_arr.append(mean_abs_error)
    return np.mean(mae_arr), mae_arr


def mase(results_df, data_df, freq_delta):
    pred_length = int(results_df.columns[-1])
    mae_arr = []
    for h in range(1,pred_length+1):
        shift_results = results_df[['ds', 'unique_id', str(h)]]
        shift_results.loc[:,'ds'] += freq_delta * h
        merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
        mean_abs_error = np.mean(np.abs(merged_results['y'] - merged_results[str(h)]))
        mae_arr.append(mean_abs_error)
    
    # naive mae
    shift_results = data_df.copy()
    shift_results.loc[:, 'ds'] -= freq_delta
    shift_results = shift_results.rename(columns={"y": "1"})
    merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
    mae_n = np.mean(np.abs(merged_results['y'] - merged_results["1"]))
    return np.mean(mae_arr) / mae_n, np.array(mae_arr) / mae_n


def tce(lower_df, upper_df, data_df, freq_delta, confidence):    
    pred_length = int(lower_df.columns[-1])
    outside_ratio = (1-confidence)/2
    tce_arr = []
    for h in range(1,pred_length+1):
        shift_lower = lower_df[['ds', 'unique_id', str(h)]]
        shift_lower.loc[:,'ds'] += freq_delta * h
        shift_upper = upper_df[['ds', 'unique_id', str(h)]]
        shift_upper.loc[:,'ds'] += freq_delta * h
        merged_upper = pd.merge(data_df, shift_upper, on=['unique_id', 'ds'], how='inner')
        merged_lower = pd.merge(data_df, shift_lower, on=['unique_id', 'ds'], how='inner')
        mean_upper_outside = np.mean(merged_upper['y'] > merged_upper[str(h)])
        mean_lower_outside = np.mean(merged_lower['y'] < merged_lower[str(h)])
        tce_arr.append(abs(outside_ratio - mean_upper_outside) + abs(outside_ratio - mean_lower_outside))
    return np.mean(tce_arr), np.array(tce_arr)


def wql(quantiles_dict, data_df, freq_delta):
    '''
    returns: weighted quantile loss, (pred_length) WQL array
    '''
    ql_arr = []
    for quantile, quantile_df in quantiles_dict.items():
        quantile_ql_arr = []
        pred_length = int(quantile_df.columns[-1])
        for h in range(1,pred_length+1):
            shift_results = quantile_df[['ds', 'unique_id', str(h)]]
            shift_results.loc[:,'ds'] += freq_delta * h
            merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
            quantile_loss = np.mean((2*(1-quantile)*(merged_results[str(h)] - merged_results['y'])*(merged_results[str(h)] >= merged_results['y'])) \
                            + (2*(quantile)*(merged_results['y'] - merged_results[str(h)])*(merged_results[str(h)] < merged_results['y'])))
            quantile_ql_arr.append(quantile_loss)
        ql_arr.append(quantile_ql_arr)

    scale = np.sum(merged_results['y'])
    wql_arr = np.array(ql_arr) / scale
    return np.sum(wql_arr), np.mean(wql_arr, axis=0)


def msis(lower_df, upper_df, data_df, freq_delta, confidence):    
    pred_length = int(lower_df.columns[-1])
    mis_arr = []
    for h in range(1,pred_length+1):
        shift_lower = lower_df[['ds', 'unique_id', str(h)]]
        shift_lower.loc[:,'ds'] += freq_delta * h
        shift_upper = upper_df[['ds', 'unique_id', str(h)]]
        shift_upper.loc[:,'ds'] += freq_delta * h
        merged_upper = pd.merge(data_df, shift_upper, on=['unique_id', 'ds'], how='inner')
        merged_lower = pd.merge(data_df, shift_lower, on=['unique_id', 'ds'], how='inner')
        mean_interval_score = np.mean( (merged_upper[str(h)] - merged_lower[str(h)]) \
                                      + confidence * (merged_lower[str(h)] - merged_lower['y']) * (merged_lower['y'] < merged_lower[str(h)]) \
                                      + confidence * (merged_upper['y'] - merged_upper[str(h)]) * (merged_upper['y'] > merged_upper[str(h)]) )
        mis_arr.append(mean_interval_score)
    
    # naive mae
    shift_results = data_df.copy()
    shift_results.loc[:, 'ds'] -= freq_delta
    shift_results = shift_results.rename(columns={"y": "1"})
    merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
    mae_n = np.mean(np.abs(merged_results['y'] - merged_results["1"]))

    return np.mean(mis_arr) / mae_n, np.array(mis_arr) / mae_n