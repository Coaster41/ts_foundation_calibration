import os
import argparse
import pandas as pd
from utils.utils import load_test_data
import importlib
import timesfm_run, chronos_run, moirai_run
lagllama_run = importlib.import_module("lag-llama_run")
import autoarima_run, nbeats_run

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
        "--model", type=str, default="", help="Model to run"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="model_ckpts/lag-llama.ckpt", help="Path to Lag-Llama model checkpoint"
    )
    parser.add_argument(
        "--season", type=int, default=1, help="season length (ARIMA)"
    )

    args = parser.parse_args()
    pred_length = args.pred_length
    context = args.context
    # dataset = args.dataset
    # forecast_date = args.forecast_date
    quantiles = [int(quantile) for quantile  in args.quantiles.split(',')]

    anomaly_dir = args.dataset
    fns = sorted(os.listdir(anomaly_dir))
    for fn in fns:
        print(fn)
        save_dir = args.save_dir + fn[:-4] + '/'
        forecast_date = pd.Timestamp('20000101') + pd.Timedelta(int(fn.split('_')[4]) + 1, unit='S')
        if args.model == 'timesfm':
            test_data, freq, unit, freq_delta = load_test_data(pred_length, context, quantiles, f'{anomaly_dir}{fn}', forecast_date)
            tfm = timesfm_run.load_model()
            timesfm_run.run_model(test_data, quantiles, pred_length, unit, freq, freq_delta, save_dir, context, tfm=tfm)
        
        if args.model == 'chronos':
            test_data, freq, unit, freq_delta = load_test_data(pred_length, context, quantiles, f'{anomaly_dir}{fn}', forecast_date)
            pipeline = chronos_run.load_model()
            chronos_run.run_model(test_data, quantiles, pred_length, unit, freq, freq_delta, save_dir, pipeline=pipeline)
            
        if args.model == 'moirai':
            test_data, freq, unit, freq_delta = load_test_data(pred_length, context, quantiles, f'{anomaly_dir}{fn}', forecast_date)
            model = moirai_run.load_model(pred_length, context)
            moirai_run.run_model(test_data, quantiles, pred_length, unit, freq, freq_delta, save_dir, context, model=model)

        if args.model == 'lag-llama':
            test_data, freq, unit, freq_delta = load_test_data(pred_length, context, quantiles, f'{anomaly_dir}{fn}', forecast_date)
            estimator = lagllama_run.load_model(pred_length, context, args.ckpt_path)
            lagllama_run.run_model(test_data, quantiles, pred_length, unit, freq, freq_delta, save_dir, args.ckpt_path, context, estimator=estimator)

        if args.model == 'autoarima':
            autoarima_run.run_model(f'{anomaly_dir}{fn}', quantiles, pred_length, None, None, None, save_dir, context, forecast_date, args.season)

        if args.model == 'nbeats':
            nbeats_run.run_model(f'{anomaly_dir}{fn}', quantiles, pred_length, None, None, None, save_dir, context, forecast_date)

        