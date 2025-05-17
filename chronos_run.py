from collections import defaultdict
import pandas as pd
import numpy as np
from chronos import ChronosPipeline
import multiprocessing

import torch
import argparse
import os
import time
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.itertools import batcher
from utils.utils import load_test_data

PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer


def run_model(test_data, quantiles, pred_length, unit, freq, freq_delta, save_dir):    
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    mean_results = []
    median_results = []
    quantile_results = [[] for _ in quantiles]
    start_time = time.time()
    id = -1
    for batch in batcher(test_data.input, batch_size=BSZ):
        context = [torch.tensor(entry["target"]) for entry in batch]
        quantile_forecasts, mean_forecasts = pipeline.predict_quantiles(
            context=context,
            prediction_length=pred_length,
            quantile_levels=[0.5, *(np.array(quantiles)/100)],
        )
        mean_forecasts = mean_forecasts.detach().cpu().numpy()
        quantile_forecasts = quantile_forecasts.detach().cpu().numpy()
        for entry, quantile_forecast, mean_forecast in zip(batch, quantile_forecasts, mean_forecasts):
            if id != entry["item_id"]:
                id = entry["item_id"]
                print(f"Run Time: {time.time()-start_time:.2f}, ID: {id}")
            start_date = entry["start"] + freq_delta * (len(entry["target"])-1) 
            mean_results.append([id, start_date, *mean_forecast])
            median_results.append([id, start_date, *quantile_forecast[:,0]])
            for i in range(len(quantiles)):
                quantile_results[i].append([id, start_date, *quantile_forecast[:,i+1]])

    print('done')

    os.makedirs(save_dir, exist_ok=True)
    columns = ['unique_id', 'ds', *range(1,pred_length+1)]
    mean_results = pd.DataFrame(mean_results, columns=columns)
    mean_results.to_csv(f"{save_dir}/mean_preds.csv")
    median_results = pd.DataFrame(median_results, columns=columns)
    median_results.to_csv(f"{save_dir}/median_preds.csv")
    for i, quantile in enumerate(quantiles):
        quantile_result = pd.DataFrame(quantile_results[i], columns=columns)
        quantile_results[i] = quantile_result
        quantile_result.to_csv(f"{save_dir}/quantile_{quantile}_preds.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a model and dataset, then make predictions."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Path to save results"
    )
    # parser.add_argument(
    #     "--freq", type=str, default="h", help="Frequency (M, W, D, h, min, s)"
    # )
    parser.add_argument(
        "--context", type=int, default=512, help="Size of context"
    )
    parser.add_argument(
        "--pred_length", type=int, default=24, help="Prediction horizon length"
    )
    parser.add_argument(
        "--quantiles", type=str, default="10,90", help="Prediction quantiles (comma delimited)"
    )
    parser.add_argument(
        "--forecast_date", type=str, default="", help="Date to start forecasting from"
    )

    args = parser.parse_args()
    pred_length = args.pred_length
    context = args.context
    dataset = args.dataset
    forecast_date = args.forecast_date
    quantiles = [int(quantile) for quantile  in args.quantiles.split(',')]

    test_data, freq, unit, freq_delta = load_test_data(pred_length, context, quantiles, dataset, forecast_date)
    run_model(test_data, quantiles, pred_length, unit, freq, freq_delta, args.save_dir)
#     os.makedirs(args.save_dir, exist_ok=True)

#     # Load dataframe and GluonTS dataset
#     df = pd.read_csv(args.dataset, index_col=0, parse_dates=['ds'])    
#     # df = pd.read_csv(args.dataset, index_col=False, parse_dates=['ds'])
#     ds = PandasDataset.from_long_dataframe(df, target="y", item_id="unique_id", timestamp='ds')
#     unit = ds.freq
#     freq_id = {"M":1, "W":1, "D":0, "h":0, "min":0, "s":0}[unit]

    
#     if args.forecast_date == "":
#         forecast_date = min(df['ds']) + pd.Timedelta(CTX, unit=unit)
#     else:
#         forecast_date = pd.Timestamp(args.forecast_date)
#     end_date = max(df['ds'])
#     total_forecast_length = (end_date-forecast_date) // pd.Timedelta(1, unit=unit)

#     _, test_template = split(
#         ds, date=pd.Period(forecast_date, freq=unit)
#     )

#     # Construct rolling window evaluation
#     test_data = test_template.generate_instances(
#         prediction_length=PDT,  # number of time steps for each prediction
#         windows=total_forecast_length-PDT,  # number of windows in rolling window evaluation
#         distance=1,  # number of time steps between each window - distance=PDT for non-overlapping windows
#         max_history=CTX,
#     )

#     # Load Model
#     pipeline = ChronosPipeline.from_pretrained(
#         "amazon/chronos-t5-small",
#         device_map="cuda",
#         torch_dtype=torch.bfloat16,
#     )

#     mean_results = []
#     median_results = []
#     quantile_results = [[] for _ in quantiles]
#     start_time = time.time()
#     for batch in batcher(test_data.input, batch_size=BSZ):
#         context = [torch.tensor(entry["target"]) for entry in batch]
#         quantile_forecasts, mean_forecasts = pipeline.predict_quantiles(
#             context=context,
#             prediction_length=PDT,
#             quantile_levels=[0.5, *(np.array(quantiles)/100)],
#         )
#         mean_forecasts = mean_forecasts.detach().cpu().numpy()
#         quantile_forecasts = quantile_forecasts.detach().cpu().numpy()
#         for entry, quantile_forecast, mean_forecast in zip(batch, quantile_forecasts, mean_forecasts):
#             if id != entry["item_id"]:
#                 id = entry["item_id"]
#                 print(f"Run Time: {time.time()-start_time:.2f}, ID: {id}")
#             start_date = entry["start"] + pd.Timedelta(len(entry["target"])-1, unit)
#             mean_results.append([id, start_date, *mean_forecast])
#             median_results.append([id, start_date, *quantile_forecast[:,0]])
#             for i in range(len(quantiles)):
#                 quantile_results[i].append([id, start_date, *quantile_forecast[:,i]])

#     print('done')

#     columns = ['unique_id', 'ds', *range(1,PDT+1)]
#     mean_results = pd.DataFrame(mean_results, columns=columns)
#     mean_results.to_csv(f"{args.save_dir}/mean_preds.csv")
#     median_results = pd.DataFrame(median_results, columns=columns)
#     median_results.to_csv(f"{args.save_dir}/median_preds.csv")
#     for i, quantile in enumerate(quantiles):
#         quantile_result = pd.DataFrame(quantile_results[i], columns=columns)
#         quantile_results[i] = quantile_result
#         quantile_result.to_csv(f"{args.save_dir}/quantile_{quantile}_preds.csv")