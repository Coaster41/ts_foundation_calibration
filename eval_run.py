import pandas as pd
import numpy as np
import time
from utils.evals import *
from collections import defaultdict

UNIT_DICT = {"amazon-google": "H", "m5": "D", "glucose": "T", "meditation": "S"}
UNIT_NUM_DICT = {"amazon-google": 1, "m5": 1, "glucose": 5, "meditation": 1}

def load_results(dataset, model):
    unit = UNIT_DICT[dataset]
    unit_num = UNIT_NUM_DICT[dataset]
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

    results_fn = f"model_results/{dataset}/{model}/median_preds.csv"
    # lower_fn = f"model_results/{dataset}/{model}/quantile_{round((1-confidence)/2 * 100)}_preds.csv"
    # upper_fn = f"model_results/{dataset}/{model}/quantile_{round((1+confidence)/2 * 100)}_preds.csv"
    data_fn = f"data/{dataset}/y_{dataset}.csv"
    results_df = pd.read_csv(results_fn, index_col=0, parse_dates=['ds'])
    # upper_df = pd.read_csv(upper_fn, index_col=0, parse_dates=['ds'])
    # lower_df = pd.read_csv(lower_fn, index_col=0, parse_dates=['ds'])
    data_df = pd.read_csv(data_fn, index_col=0, parse_dates=['ds'])
    freq_delta = pd.Timedelta(unit_num, unit=unit)
        
    quantiles_df = []
    for quantile in quantiles:
        quantile_fn = f"model_results/{dataset}/{model}/quantile_{round(quantile * 100)}_preds.csv"
        quantiles_df.append(pd.read_csv(quantile_fn, index_col=0, parse_dates=['ds']))
    quantiles_df.insert(len(quantiles_df)//2, results_df)
    quantiles.insert(len(quantiles)//2, 0.5)
    quantiles_dict = dict(zip(quantiles, quantiles_df))
    return data_df, results_df, freq_delta, quantiles_dict


if __name__ == "__main__":
    datasets = ["amazon-google", "m5", "glucose", "meditation"]
    # models = ["timesfm", "moirai", "chronos", "lag-llama", "nbeats", "autoarima"]
    models = ["lag-llama-context", "nbeats-context"]
    results = []
    # confidence = 0.8
    confidences = [0.8, 0.6, 0.4, 0.2, 0]
    start_time = time.time()
    max_ids = 0

    for dataset in datasets:
        for model in models:
            print(f"Time: {time.time()-start_time:.4f}\t{dataset} {model}")
            data_df, results_df, freq_delta, quantiles_dict = load_results(dataset, model)
            mase_avg, mase_arr = mase(results_df, data_df, freq_delta)
            results.append([dataset, model, 'mase', mase_avg, *mase_arr])
            wql_avg, wql_arr = wql(quantiles_dict, data_df, freq_delta)
            results.append([dataset, model, 'wql', wql_avg, *wql_arr])
            pce_avg, pce_arr = pce(quantiles_dict, data_df, freq_delta)
            results.append([dataset, model, 'pce', pce_avg, *pce_arr])
            mpiqr_avg, mpiqr_arr = mpiqr_mean(quantiles_dict, data_df, confidences)
            results.append([dataset, model, 'mpiqr', mpiqr_avg, *mpiqr_arr])
            for confidence in confidences:
                upper_df = quantiles_dict[round(0.5 + confidence/2, 1)]
                lower_df = quantiles_dict[round(0.5 - confidence/2, 1)]
                tce_avg, tce_arr = tce(lower_df, upper_df, data_df, freq_delta, confidence)
                results.append([dataset, model, f'tce_{round(confidence*100)}', tce_avg, *tce_arr])
                msiw_avg, msiw_arr = msiw(lower_df, upper_df, data_df)
                results.append([dataset, model, f'msiw_{round(confidence*100)}', msiw_avg, *msiw_arr])
                msis_avg, msis_arr = msis(lower_df, upper_df, data_df, freq_delta, confidence)
                results.append([dataset, model, f'msis_{round(confidence*100)}', msis_avg, *msis_arr])
                mpiqr_avg, mpiqr_arr = mpiqr(lower_df, upper_df, data_df, confidence)
                results.append([dataset, model, f'mpiqr_{round(confidence*100)}', mpiqr_avg, *mpiqr_arr])
            df = pd.DataFrame(results, columns=['dataset', 'model', 'metric', 'avg_result', *[str(h) for h in range(1,49)]])
            df.index += 576
            df.to_csv('model_results/metric_results_test1.csv')
    df = pd.DataFrame(results, columns=['dataset', 'model', 'metric', 'avg_result', *[str(h) for h in range(1,49)]])
    # df.to_csv('model_results/metric_results.csv')

    # groupby unique_id
    results = []
    for dataset in datasets:
        for model in models:
            print(f"Time-Series: {time.time()-start_time:.4f}\t{dataset} {model}")
            data_df, results_df, freq_delta, quantiles_dict = load_results(dataset, model)
            grouped_data_df = data_df.groupby('unique_id')
            grouped_results_df = results_df.groupby('unique_id')
            grouped_quantiles_dict = defaultdict(dict)
            for quantile, quantile_df in quantiles_dict.items():
                grouped_quantile_df = quantile_df.groupby('unique_id')
                for unique_id, grouped_q in grouped_quantile_df:
                    grouped_quantiles_dict[unique_id][quantile] = grouped_q
            max_ids = max(max_ids, len(grouped_data_df))

            # mase_arr, tce_arr, msis_arr, wql_arr, pce_arr, msiw_arr = list(), list(), list(), list(), list(), list()
            mase_arr, wql_arr, pce_arr, mpiqr_mean_arr = list(), list(), list(), list()
            msis_arr = [[] for _ in range(len(confidences))]
            tce_arr = [[] for _ in range(len(confidences))]
            msiw_arr = [[] for _ in range(len(confidences))]
            mpiqr_arr = [[] for _ in range(len(confidences))]
            for (unique_id, data_df), (_, results_df), (_, quantiles_dict) in \
                zip(grouped_data_df, grouped_results_df, grouped_quantiles_dict.items()):
                mase_avg, _ = mase(results_df, data_df, freq_delta)
                mase_arr.append(mase_avg)
                wql_avg, _ = wql(quantiles_dict, data_df, freq_delta)
                wql_arr.append(wql_avg)
                pce_avg, _ = pce(quantiles_dict, data_df, freq_delta)
                pce_arr.append(pce_avg)
                mpiqr_avg, _ = mpiqr_mean(quantiles_dict, data_df, confidences)
                mpiqr_mean_arr.append(mpiqr_avg)
                for i, confidence in enumerate(confidences):
                    upper_df = quantiles_dict[round(0.5 + confidence/2, 1)]
                    lower_df = quantiles_dict[round(0.5 - confidence/2, 1)]
                    tce_avg, _ = tce(lower_df, upper_df, data_df, freq_delta, confidence)
                    tce_arr[i].append(tce_avg)
                    msis_avg, _ = msis(lower_df, upper_df, data_df, freq_delta, confidence)
                    msis_arr[i].append(msis_avg)
                    msiw_avg, _ = msiw(lower_df, upper_df, data_df)
                    msiw_arr[i].append(msiw_avg)
                    mpiqr_avg, _ = mpiqr(lower_df, upper_df, data_df, confidence)
                    mpiqr_arr[i].append(mpiqr_avg)
                    
            results.append([dataset, model, 'pce', *pce_arr])
            results.append([dataset, model, 'mase', *mase_arr])
            results.append([dataset, model, 'wql', *wql_arr])
            results.append([dataset, model, 'mpiqr', *mpiqr_mean_arr])

            results.extend([[dataset, model, f'tce_{round(confidence*100)}', *tce_arr[i]] for i, confidence in enumerate(confidences)])
            results.extend([[dataset, model, f'msis_{round(confidence*100)}', *msis_arr[i]] for i, confidence in enumerate(confidences)])
            results.extend([[dataset, model, f'msiw_{round(confidence*100)}', *msiw_arr[i]] for i, confidence in enumerate(confidences)])
            results.extend([[dataset, model, f'mpiqr_{round(confidence*100)}', *mpiqr_arr[i]] for i, confidence in enumerate(confidences)])
            df = pd.DataFrame(results, columns=['dataset', 'model', 'metric', *[str(h) for h in range(1,max_ids+1)]])
            df.index += 576
            df.to_csv('model_results/metric_results_unique_id_test1.csv')
    df = pd.DataFrame(results, columns=['dataset', 'model', 'metric', *[str(h) for h in range(1,max_ids+1)]])
    # df.to_csv('model_results/metric_results_unique_id.csv')