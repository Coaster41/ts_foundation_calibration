from collections import defaultdict
import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import DistributionLoss

import torch
import argparse
import os
import time
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.itertools import batcher
from lightning.pytorch.plugins.environments import SLURMEnvironment
import logging

class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        keywords = ['available:', 'CUDA', 'LOCAL_RANK:', 'SLURM']
        return not any(keyword in record.getMessage() for keyword in keywords)
    
logging.getLogger('pytorch_lightning.utilities.rank_zero').addFilter(IgnorePLFilter())
logging.getLogger('pytorch_lightning.accelerators.cuda').addFilter(IgnorePLFilter())
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer


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
    PDT = args.pred_length
    CTX = args.context
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

    train_df = df.loc[df['ds'] <= forecast_date]
    test_df = df.loc[df['ds'] > forecast_date].reset_index(drop=True)
    train_dataset, test_template = split(
        ds, date=pd.Period(forecast_date, freq=unit)
    )

    # Construct rolling window evaluation
    test_data = test_template.generate_instances(
        prediction_length=PDT,  # number of time steps for each prediction
        windows=total_forecast_length-PDT,  # number of windows in rolling window evaluation
        distance=1,  # number of time steps between each window - distance=PDT for non-overlapping windows
        max_history=CTX,
    )

    # Load Model
    model = NBEATS(h=PDT, input_size=CTX,
                    loss=DistributionLoss(distribution='Poisson', level=quantiles[len(quantiles)//2:]),
                    stack_types = ['identity', 'trend', 'seasonality'],
                    max_steps=100,
                    val_check_steps=10,
                    early_stop_patience_steps=2)
    
    model.trainer_kwargs['logger'] = False
    model.trainer_kwargs['enable_progress_bar'] = False
    fcst = NeuralForecast(
        models=[model],
        freq=unit
    )
    
    start_time = time.time()
    fcst.fit(df=train_df, val_size=PDT*2)
    print(f'Finshed fitting in {time.time()-start_time:.2f}')
    
    forecast_cols = ["NBEATS", "NBEATS-median",  \
                        *[f"NBEATS-lo-{quantile}" for quantile in quantiles[len(quantiles)//2:]], \
                        *[f"NBEATS-hi-{quantile}" for quantile in quantiles[len(quantiles)//2:]]]
    file_names = ["mean", "median", \
                        *[f"quantile_{100-quantile}_preds" for quantile in quantiles[len(quantiles)//2:]], \
                        *[f"quantile_{quantile}_preds" for quantile in quantiles[len(quantiles)//2:]]] 
    model_results = defaultdict(list)
    for last_observed in pd.date_range(start=args.forecast_date, end=end_date):
        forecast_df = fcst.predict(df=(df.loc[df['ds']<=last_observed]))
        for forecast_col, file_name in zip(forecast_cols, file_names):
            forecast_result = pd.DataFrame(forecast_df[['unique_id', forecast_col]].groupby('unique_id')[forecast_col].agg(list), 
                                            columns=[forecast_col])
            forecast_result[list(range(1,PDT+1))] = pd.DataFrame(forecast_result[forecast_col].tolist(), 
                                                                index=forecast_result.index)
            forecast_result.drop(columns=[forecast_col], inplace=True)
            forecast_result.insert(0, 'ds', last_observed)
            model_results[file_name].append(forecast_result)
    
    for file_name in file_names:
        forecast_result = pd.concat(model_results[file_name], ignore_index=True)
        forecast_result.to_csv(f"{args.save_dir}/{file_name}.csv")

    print(f'Done in {time.time()-start_time:.2f}')