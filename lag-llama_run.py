from collections import defaultdict
import pandas as pd
import numpy as np

from itertools import islice

import torch
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.evaluation import make_evaluation_predictions

import multiprocessing

import torch
import argparse
import os
import time
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.itertools import batcher

PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer
num_samples = 100


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
    parser.add_argument(
        "--ckpt_path", type=str, default="model_ckpts/lag-llama.ckpt", help="Path to model checkpoint"
    )

    args = parser.parse_args()
    PDT = args.pred_length
    CTX = args.context
    ckpt_path = args.ckpt_path
    gpu_num = (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS"))
    device = torch.device(f"cuda:{gpu_num}") if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    quantiles = [int(quantile) for quantile  in args.quantiles.split(',')]
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataframe and GluonTS dataset
    df = pd.read_csv(args.dataset, index_col=0, parse_dates=['ds'])    
    # df = pd.read_csv(args.dataset, index_col=False, parse_dates=['ds'])
    ds = PandasDataset.from_long_dataframe(df, target="y", item_id="unique_id", timestamp='ds')
    unit = ds.freq
    freq_id = {"M":1, "W":1, "D":0, "h":0, "min":0, "s":0}[unit]

    
    if args.forecast_date == "":
        forecast_date = min(df['ds']) + pd.Timedelta(CTX, unit=unit)
    else:
        forecast_date = pd.Timestamp(args.forecast_date)
    end_date = max(df['ds'])
    total_forecast_length = (end_date-forecast_date) // pd.Timedelta(1, unit=unit)

    _, test_template = split(
        ds, date=pd.Period(forecast_date, freq=unit)
    )

    # Construct rolling window evaluation
    test_data = test_template.generate_instances(
        prediction_length=PDT,  # number of time steps for each prediction
        windows=total_forecast_length-PDT,  # number of windows in rolling window evaluation
        distance=1,  # number of time steps between each window - distance=PDT for non-overlapping windows
        max_history=CTX,
    )
    print(total_forecast_length-PDT)

    # Load Model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (CTX + PDT) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=PDT,
        context_length=CTX, # Lag-Llama was trained with a context length of 32, but can work with any context length

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=None, # ???

        batch_size=BSZ,
        num_parallel_samples=100,
        device=device,
    )
    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    forecasts = predictor.predict(test_data.input)
    forecast_it = iter(forecasts)

    mean_results = []
    median_results = []
    quantile_results = [[] for _ in quantiles]
    start_time = time.time()
    for i, (forecast) in enumerate(forecast_it):
        start_date = forecast.index[0] - pd.Timedelta(1, unit)
        # print(f"time: {time.time()-start_time:.2f} date: {start_date} id: {forecast.item_id}")
        mean_results.append([forecast.item_id, start_date, *np.mean(forecast.samples, axis=0)])
        median_results.append([forecast.item_id, start_date, *np.median(forecast.samples, axis=0)])
        for i, quantile in enumerate(quantiles):
            quantile_results[i].append([forecast.item_id, start_date, \
                                        *np.quantile(forecast.samples, q=quantile/100, axis=0)])

    print('done')

    columns = ['unique_id', 'ds', *range(1,PDT+1)]
    mean_results = pd.DataFrame(mean_results, columns=columns)
    mean_results.to_csv(f"{args.save_dir}/mean_preds.csv")
    median_results = pd.DataFrame(median_results, columns=columns)
    median_results.to_csv(f"{args.save_dir}/median_preds.csv")
    for i, quantile in enumerate(quantiles):
        quantile_result = pd.DataFrame(quantile_results[i], columns=columns)
        quantile_results[i] = quantile_result
        quantile_result.to_csv(f"{args.save_dir}/quantile_{quantile}_preds.csv")