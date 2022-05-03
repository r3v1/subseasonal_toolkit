import os
from datetime import datetime

import numpy as np
import pandas as pd
from subseasonal_data import data_loaders
from subseasonal_data.utils import get_measurement_variable

from subseasonal_toolkit.utils.eval_util import get_target_dates, mean_rmse_to_score, save_metric
from subseasonal_toolkit.utils.experiments_util import get_first_year, get_forecast_delta
from subseasonal_toolkit.utils.general_util import printf
from subseasonal_toolkit.utils.models_util import (get_submodel_name, start_logger, log_params, get_forecast_filename,
                                                   save_forecasts)

model_name = "deb_cfsv2"
# Otherwise, specify arguments interactively
gt_id = "us_precip_1.5x1.5"
horizon = "34w"
target_dates = "std_ecmwf"
first_year = 1999
last_year = 2010
if horizon == "34w":
    first_lead = 15
    last_lead = 15
elif horizon == "56w":
    first_lead = 29
    last_lead = 29

if __name__ == "__main__":
    #
    # Choose regression parameters
    #
    # Record standard settings of these parameters
    if gt_id.endswith("1.5x1.5"):
        prefix = "iri_cfsv2"
    else:
        prefix = "subx_cfsv2"
    if "tmp2m" in gt_id:
        base_col = prefix + '_tmp2m'
    elif "precip" in gt_id:
        base_col = prefix + '_precip'

    #
    # Process model parameters
    #

    # Get list of target date objects
    target_date_objs = pd.Series(get_target_dates(date_str=target_dates, horizon=horizon))

    # Identify measurement variable name
    measurement_variable = get_measurement_variable(gt_id)  # 'tmp2m' or 'precip'

    # Column name for ground truth
    gt_col = measurement_variable

    LAST_SAVE_YEAR = get_first_year(prefix)  # Don't save forecasts for years earlier than LAST_SAVE_YEAR

    # Record model and submodel names
    submodel_name = get_submodel_name(
        model_name,
        first_year=first_year, last_year=last_year,
        first_lead=first_lead, last_lead=last_lead)

    # Save output to log file
    logger = start_logger(model=model_name, submodel=submodel_name, gt_id=gt_id,
                          horizon=horizon, target_dates=target_dates)
    # Store parameter values in log
    params_names = ['gt_id', 'horizon', 'target_dates',
                    'first_year', 'last_year',
                    'first_lead', 'last_lead',
                    'base_col'
                    ]
    params_values = [eval(param) for param in params_names]
    log_params(params_names, params_values)

    # Load and process CFSv2 data
    printf("Loading cfsv2 data and averaging leads")
    # Choose data shift based on horizon
    base_shift = get_forecast_delta(horizon)

    # Carga la máscara de US
    mask = None
    if gt_id.startswith("us_"):
        suffix = "-us"
    else:
        suffix = ""
    if gt_id.endswith("1.5x1.5"):
        suffix += "1_5"
    else:
        mask = data_loaders.get_us_mask()

    data = data_loaders.get_forecast(prefix + "-" + measurement_variable + suffix,
                                     # mask_df=mask,
                                     shift=base_shift)
    cols = [prefix + "_" + gt_id.split("_")[1] + "-{}.5d_shift{}".format(col, base_shift)
            for col in range(first_lead, last_lead + 1)]
    data[base_col] = data[cols].mean(axis=1)

    printf('Pivoting dataframe to have one row per start_date')
    data = data[['lat', 'lon', 'start_date', base_col]].set_index(['lat', 'lon', 'start_date']).unstack(['lat', 'lon'])

    # Load ground truth
    gt = data_loaders.get_ground_truth(gt_id).loc[:, ['lat', 'lon', 'start_date', gt_col]]
    printf('Pivoting ground truth to have one row per start_date')
    gt = gt.loc[gt.start_date.isin(data.index), ['lat', 'lon', 'start_date', gt_col]].set_index(
        ['lat', 'lon', 'start_date']).unstack(['lat', 'lon'])
    printf("Merging ground truth")
    data = data.join(gt, how="left")
    del gt

    # Identify the month-day combination for each date treating 2/29 as 2/28
    monthdays = pd.Series([(d.month, d.day) if d.month != 2 or d.day != 29
                           else (2, 28) for d in data.index], index=data.index)

    # Compute debiasing correction
    printf('Compute debiasing correction (ground-truth - base prediction) by month-day combination')
    debias = (data[gt_col] - data[base_col])  # Calcula el bias respecto a cada celda y cada paso día
    debias = debias[(debias.index >= str(first_year)) & (debias.index <= str(last_year))]  # Filtra el conjunto train
    debias = debias.groupby(by=monthdays[debias.index]).mean()  # Obtiene el bias promedio por mes y día

    # Make predictions for each target date
    printf('Creating dataframe to store performance')
    rmses = pd.Series(index=target_date_objs, dtype=np.float64)
    printf('Forming debiased predictions for target dates')
    # Form predictions for target dates in data matrix
    valid_targets = data.index.intersection(target_date_objs)
    target_monthdays = monthdays.loc[valid_targets]

    # Aquí aplica la corrección sumándole el bias promedio (esta es la predicción)
    preds = data.loc[valid_targets, base_col] + debias.loc[target_monthdays].values

    preds.index.name = "start_date"
    # Order valid targets by day of week
    valid_targets = valid_targets[valid_targets.weekday.argsort(kind='stable')]
    for target_date_obj in valid_targets:
        # Skip if forecast already produced for this target
        target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
        forecast_file = get_forecast_filename(
            model=model_name, submodel=submodel_name,
            gt_id=gt_id, horizon=horizon,
            target_date_str=target_date_str)
        if os.path.isfile(forecast_file):
            printf(f"prior forecast exists for target={target_date_obj}")
            pred = pd.read_hdf(forecast_file).set_index(['lat', 'lon']).pred
        else:
            printf(f'Processing {model_name} forecast for {target_date_obj}')
            # Add correction to base prediction
            pred = preds.loc[target_date_obj, :]
            # Save prediction to file in standard format
            if target_date_obj.year >= LAST_SAVE_YEAR:
                save_forecasts(
                    preds.loc[[target_date_obj], :].unstack().rename("pred").reset_index(),
                    model=model_name, submodel=submodel_name,
                    gt_id=gt_id, horizon=horizon,
                    target_date_str=target_date_str)
        # Evaluate and store error if we have ground truth data
        if target_date_obj in data.index:
            rmse = np.sqrt(np.square(pred - data.loc[target_date_obj, gt_col]).mean())
            rmses.loc[target_date_obj] = rmse
            printf("-rmse: {}, score: {}".format(rmse, mean_rmse_to_score(rmse)))
            mean_rmse = rmses.mean()
            printf("-mean rmse: {}, running score: {}".format(mean_rmse, mean_rmse_to_score(mean_rmse)))

    printf("Save rmses in standard format")
    rmses = rmses.sort_index().reset_index()
    rmses.columns = ['start_date', 'rmse']
    save_metric(rmses, model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon,
                target_dates=target_dates, metric="rmse")
