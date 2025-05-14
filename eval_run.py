import pandas as pd
import numpy as np
import time
from utils.evals import *

UNIT_DICT = {"amazon-google": "H", "m5": "D", "glucose": "T", "meditation": "S"}
UNIT_NUM_DICT = {"amazon-google": 1, "m5": 1, "glucose": 5, "meditation": 1}

def load_results(dataset, model, confidence=0.8):
    # dataset = 'amazon-google'
    # model = 'timesfm'
    # confidence = 0.80

    unit = UNIT_DICT[dataset]
    unit_num = UNIT_NUM_DICT[dataset]
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

    results_fn = f"model_results/{dataset}/{model}/median_preds.csv"
    lower_fn = f"model_results/{dataset}/{model}/quantile_{round((1-confidence)/2 * 100)}_preds.csv"
    upper_fn = f"model_results/{dataset}/{model}/quantile_{round((1+confidence)/2 * 100)}_preds.csv"
    data_fn = f"data/{dataset}/y_{dataset}.csv"
    results_df = pd.read_csv(results_fn, index_col=0, parse_dates=['ds'])
    upper_df = pd.read_csv(upper_fn, index_col=0, parse_dates=['ds'])
    lower_df = pd.read_csv(lower_fn, index_col=0, parse_dates=['ds'])
    data_df = pd.read_csv(data_fn, index_col=0, parse_dates=['ds'])
    freq_delta = pd.Timedelta(unit_num, unit=unit)
        
    quantiles_df = []
    for quantile in quantiles:
        quantile_fn = f"model_results/{dataset}/{model}/quantile_{round(quantile * 100)}_preds.csv"
        quantiles_df.append(pd.read_csv(quantile_fn, index_col=0, parse_dates=['ds']))
    quantiles_df.insert(len(quantiles_df)//2, results_df)
    quantiles.insert(len(quantiles)//2, 0.5)
    quantiles_dict = dict(zip(quantiles, quantiles_df))
    return data_df, results_df, upper_df, lower_df, freq_delta, quantiles_dict


if __name__ == "__main__":
    datasets = ["amazon-google", "m5", "glucose", "meditation"]
    models = ["timesfm", "moirai", "chronos", "lag-llama", "nbeats", "autoarima"]
    results = []
    confidence = 0.8
    start_time = time.time()
    for dataset in datasets:
        for model in models:
            print(f"Time: {time.time()-start_time:.4f}\t{dataset} {model}")
            data_df, results_df, upper_df, lower_df, freq_delta, quantiles_dict = load_results(dataset, model, confidence)
            mase_avg, mase_arr = mase(results_df, data_df, freq_delta)
            results.append([dataset, model, 'mase', mase_avg, *mase_arr])
            tce_avg, tce_arr = tce(lower_df, upper_df, data_df, freq_delta, confidence)
            results.append([dataset, model, 'tce', tce_avg, *tce_arr])
            wql_avg, wql_arr = wql(quantiles_dict, data_df, freq_delta)
            results.append([dataset, model, 'wql', wql_avg, *wql_arr])
            msis_avg, msis_arr = msis(lower_df, upper_df, data_df, freq_delta, confidence)
            results.append([dataset, model, 'msis', msis_avg, *tce_arr])
    df = pd.DataFrame(results, columns=['dataset', 'model', 'metric', 'avg_result', *[str(h) for h in range(1,49)]])
    df.to_csv('model_results/metric_results.csv')