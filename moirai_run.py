import pandas as pd
import numpy as np
import torch
import argparse
import os
import time
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from utils.utils import load_test_data

MODEL = "moirai"  # model name: choose from {'moirai', 'moirai-moe'}

# SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
# PDT = 24  # prediction length: any positive integer
# CTX = 200  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer

def run_model(test_data, quantiles, PDT, unit, freq, freq_delta, save_dir, CTX):

    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-small"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=32,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    
    predictor = model.create_predictor(batch_size=BSZ)
    forecasts = predictor.predict(test_data.input)

    input_it = iter(test_data.input)
    label_it = iter(test_data.label)
    forecast_it = iter(forecasts)


    # print(len(forecast_it))
    mean_results = []
    median_results = []
    quantile_results = [[] for _ in quantiles]
    start_time = time.time()
    for i, (input, label, forecast) in enumerate(zip(input_it, label_it, forecast_it)):
        start_date = forecast.index[0] - freq_delta
        mean_results.append([forecast.item_id, start_date, *np.mean(forecast.samples, axis=0)])
        median_results.append([forecast.item_id, start_date, *np.median(forecast.samples, axis=0)])
        for i, quantile in enumerate(quantiles):
            quantile_results[i].append([forecast.item_id, start_date, \
                                        *np.quantile(forecast.samples, q=quantile/100, axis=0)])
    print("done")
    os.makedirs(save_dir, exist_ok=True)

    columns = ['unique_id', 'ds', *range(1,PDT+1)]
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
    run_model(test_data, quantiles, pred_length, unit, freq, freq_delta, args.save_dir, context)

    # args = parser.parse_args()
    # PDT = args.pred_length
    # CTX = args.context
    # quantiles = [int(quantile) for quantile  in args.quantiles.split(',')]
    # os.makedirs(args.save_dir, exist_ok=True)

    # # Load dataframe and GluonTS dataset
    # df = pd.read_csv(args.dataset, index_col=0, parse_dates=['ds'])    
    # # df = pd.read_csv(args.dataset, index_col=False, parse_dates=['ds'])
    # ds = PandasDataset.from_long_dataframe(df, target="y", item_id="unique_id", timestamp='ds')
    # unit = ds.freq

    
    # if args.forecast_date == "":
    #     forecast_date = min(df['ds']) + pd.Timedelta(CTX, unit=unit)
    # else:
    #     forecast_date = pd.Timestamp(args.forecast_date)
    # end_date = max(df['ds'])
    # total_forecast_length = (end_date-forecast_date) // pd.Timedelta(1, unit=unit)

    # _, test_template = split(
    #     ds, date=pd.Period(forecast_date, freq=unit)
    # )

    # # Construct rolling window evaluation
    # test_data = test_template.generate_instances(
    #     prediction_length=PDT,  # number of time steps for each prediction
    #     windows=total_forecast_length-PDT,  # number of windows in rolling window evaluation
    #     distance=1,  # number of time steps between each window - distance=PDT for non-overlapping windows
    # )
    # print(PDT, total_forecast_length-PDT)

    # model = MoiraiForecast(
    #     module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-small"),
    #     prediction_length=PDT,
    #     context_length=CTX,
    #     patch_size=32,
    #     num_samples=100,
    #     target_dim=1,
    #     feat_dynamic_real_dim=0,
    #     past_feat_dynamic_real_dim=0,
    # )
    
    # predictor = model.create_predictor(batch_size=BSZ)
    # forecasts = predictor.predict(test_data.input)

    # input_it = iter(test_data.input)
    # label_it = iter(test_data.label)
    # forecast_it = iter(forecasts)


    # # print(len(forecast_it))
    # mean_results = []
    # median_results = []
    # quantile_results = [[] for _ in quantiles]
    # start_time = time.time()
    # for i, (input, label, forecast) in enumerate(zip(input_it, label_it, forecast_it)):
    #     start_date = forecast.index[0] - pd.Timedelta(1, unit)
    #     mean_results.append([forecast.item_id, start_date, *np.mean(forecast.samples, axis=0)])
    #     median_results.append([forecast.item_id, start_date, *np.median(forecast.samples, axis=0)])
    #     for i, quantile in enumerate(quantiles):
    #         quantile_results[i].append([forecast.item_id, start_date, \
    #                                     *np.quantile(forecast.samples, q=quantile/100, axis=0)])
    # print("done")

    # columns = ['unique_id', 'ds', *range(1,PDT+1)]
    # mean_results = pd.DataFrame(mean_results, columns=columns)
    # mean_results.to_csv(f"{args.save_dir}/mean_preds.csv")
    # median_results = pd.DataFrame(median_results, columns=columns)
    # median_results.to_csv(f"{args.save_dir}/median_preds.csv")
    # for i, quantile in enumerate(quantiles):
    #     quantile_result = pd.DataFrame(quantile_results[i], columns=columns)
    #     quantile_results[i] = quantile_result
    #     quantile_result.to_csv(f"{args.save_dir}/quantile_{quantile}_preds.csv")