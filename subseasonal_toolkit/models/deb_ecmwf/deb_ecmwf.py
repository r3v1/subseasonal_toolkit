# Predicts outcomes using cfsv2++
#
# Example usages:
#   python src/models/deb_ecmwf/batch_predict.py contest_tmp2m 34w -t std_val -i True -y all -m None
#   python src/models/deb_ecmwf/batch_predict.py contest_precip 34w -t std_val -i True -y 20 -m 56
#   python src/models/deb_ecmwf/batch_predict.py contest_precip 34w -t std_val -i True -y all -m 56 -d 35
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 12w, 34w, or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --fit_intercept (-i): if "True" fits intercept to debias
#     ecmwf predictions; if "False" does not fit intercept; (default: "False")
#   --train_years (-y): number of years to use in training ("all" for all years
#     or positive integer); (default: "all")
#   --margin_in_days (-m): number of month-day combinations on either side of
#     the target combination to include; set to 0 to include only target
#     month-day combo; set to "None" to include entire year; (default: "None")
#   --first_day (-fd): first available daily ecmwf forecast (1 or greater) to average
#   --last_day (-ld): last available daily ecmwf forecast (first_day or greater) to average
#   --loss (-l): loss function: mse, rmse, skill, or ssm (default: "mse")
#   --first_lead (-fl): first ecmwf lead to average into forecast (0-29) (default: 0)
#   --last_lead (-ll): last ecmwf lead to average into forecast (0-29) (default: 29)

import os
from argparse import ArgumentParser
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from subseasonal_data import data_loaders
from subseasonal_data.utils import get_measurement_variable

from subseasonal_toolkit.models.deb_ecmwf.ecmwf_utils import geometric_median, ssm
from subseasonal_toolkit.utils.eval_util import get_target_dates, mean_rmse_to_score, save_metric
from subseasonal_toolkit.utils.experiments_util import (get_id_name, get_th_name, get_start_delta,
                                                        get_forecast_delta)
from subseasonal_toolkit.utils.general_util import printf
from subseasonal_toolkit.utils.models_util import (get_submodel_name, start_logger, log_params, get_forecast_filename,
                                                   save_forecasts)

if __name__ == "__main__":
    #
    # Specify model parameters
    #
    model_name = "deb_ecmwf"

    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon
    parser.add_argument('--target_dates', '-t', default="std_test")
    parser.add_argument('--train_years', '-y', default=20,
                        help="Number of years to use in debiasing (integer)")
    parser.add_argument('--loss', '-l', default="mse",
                        help="loss function: mse, rmse, skill, or ssm")
    parser.add_argument('--first_lead', '-fl', default=0,
                        help="first ecmwf lead to average into forecast (0-29)")
    parser.add_argument('--last_lead', '-ll', default=29,
                        help="last ecmwf lead to average into forecast (0-29)")
    parser.add_argument('--forecast_with', '-fw', default="c",
                        help="Generate forecast using the perturbed (p) or control (c) ECMWF forecast  (or p+c for both).")
    parser.add_argument('--debias_with', '-dw', default="c",
                        help="Debias using the perturbed (p) or control (c) reforecast (or p+c for both).")

    args, opt = parser.parse_known_args()

    # Assign variables
    gt_id = get_id_name(args.pos_vars[0])  # "contest_precip" or "contest_tmp2m"
    horizon = get_th_name(args.pos_vars[1])  # "12w", "34w", or "56w"
    target_dates = args.target_dates
    train_years = int(args.train_years)
    loss = args.loss
    first_lead = int(args.first_lead)
    last_lead = int(args.last_lead)
    debias_with = args.debias_with
    forecast_with = args.forecast_with

    """ 
    Choose regression parameters and record standard settings
    of these parameters
    """
    x_cols = ['zeros']
    if "tmp2m" in gt_id:
        base_col = 'ecmwf_tmp2m'
    elif "precip" in gt_id:
        base_col = 'ecmwf_precip'
    group_by_cols = ['lat', 'lon']

    """ 
    Process model parameters"""

    # Get list of target date objects
    target_date_objs = pd.Series(get_target_dates(date_str=target_dates, horizon=horizon))

    # Identify measurement variable name
    measurement_variable = get_measurement_variable(gt_id)  # 'tmp2m' or 'precip'

    # Column names for gt_col, clim_col and anom_col
    gt_col = measurement_variable
    clim_col = measurement_variable + "_clim"
    anom_col = get_measurement_variable(gt_id) + "_anom"  # 'tmp2m_anom' or 'precip_anom'

    # Store delta between target date and forecast issuance date
    forecast_delta = timedelta(days=get_start_delta(horizon, gt_id))

    # Don't save forecasts for years earlier than LAST_SAVE_YEAR
    LAST_SAVE_YEAR = 2015  # get_first_year(model_name)

    # Record model and submodel names
    submodel_name = get_submodel_name(
        model_name, train_years=train_years, loss=loss,
        first_lead=first_lead, last_lead=last_lead,
        forecast_with=forecast_with, debias_with=debias_with)
    print(submodel_name)

    # Save output to log file
    logger = start_logger(model=model_name, submodel=submodel_name, gt_id=gt_id,
                          horizon=horizon, target_dates=target_dates)
    # Store parameter values in log
    params_names = ['gt_id', 'horizon', 'target_dates', 'train_years', 'loss',
                    'first_lead', 'last_lead', 'forecast_with', 'debias_with',
                    'base_col', 'x_cols', 'group_by_cols']
    params_values = [eval(param) for param in params_names]
    log_params(params_names, params_values)

    # Select estimator based on loss
    # TODO: do we want to do this? Or always use arithmatic mean?
    if loss == "rmse":
        estimator = geometric_median
    elif loss == "ssm":
        estimator = ssm
    else:
        estimator = np.mean

    # Load and process data
    printf("Loading ecmwf data")
    # Choose data shift based on horizon and first day to be averaged
    base_shift = get_forecast_delta(horizon)

    if gt_id.startswith('us_'):
        # TODO: should remove sync=False flags when we upload the data to azure
        """Load forecast and reforecast data."""
        cols = ["iri_ecmwf_" + gt_id.split("_")[1] + "-{}.5d_shift{}".format(col, base_shift)
                for col in range(first_lead, last_lead + 1)]
        load_names = ["cf-forecast", "pf-forecast", "cf-reforecast", "pf-reforecast"]
        load_data = {}
        for l in load_names:
            print("Loading data.")
            load_data[l] = data_loaders.get_forecast(f"ecmwf-{measurement_variable}-us1_5-{l}",
                                                     mask_df=data_loaders.get_us_mask(
                                                         fname="us_1_5_mask.nc",
                                                         sync=False),
                                                     shift=base_shift,
                                                     sync=True)

            # Undo-shifting of model_issuance_date so that it matches the start date
            # TODO: this is otherwise very hard to do via the shift_df funtion, but it
            #   would be nice to have a better way of doing this
            if "reforecast" in l:
                load_data[l]['model_issuance_date'] = load_data[l][f'model_issuance_date_shift{base_shift}'] \
                                                      + timedelta(days=base_shift)

            printf("Averaging leads for forecast and debias data.")
            load_data[l][base_col] = load_data[l][cols].mean(axis=1)

            printf('Pivoting dataframe to have one row per start_date')
            if "reforecast" in l:
                load_data[l] = load_data[l][
                    ['lat', 'lon', 'start_date', 'model_issuance_date', base_col]].set_index(
                    ['lat', 'lon', 'start_date', 'model_issuance_date']).unstack(['lat', 'lon'])
            else:
                load_data[l] = load_data[l][['lat', 'lon', 'start_date', base_col]].set_index(
                    ['lat', 'lon', 'start_date']).unstack(['lat', 'lon'])

    else:
        raise ValueError("ECMWF model only configured for US gt_ids for now.")

    # Average control and perturbed for both forecast and reforecast
    print("Averaging control and perturbed forecast data.")
    for forecast_type in ["forecast", "reforecast"]:
        # Get intersection of shared forecast dates
        shared_dates = load_data[f"cf-{forecast_type}"].index.intersection(load_data[f"pf-{forecast_type}"].index)
        load_data[f"cf-{forecast_type}"] = load_data[f"cf-{forecast_type}"].loc[shared_dates]
        load_data[f"pf-{forecast_type}"] = load_data[f"pf-{forecast_type}"].loc[shared_dates]

        if forecast_type == "forecast":
            if forecast_with == "p":
                wp = 1.0
                wf = 0.0
            elif forecast_with == "c":
                wp = 0.0
                wf = 1.0
            else:
                wp = 50. / 51.
                wf = 1. / 51.

            forecast_data = wf * load_data[f"cf-{forecast_type}"] + wp * load_data[f"pf-{forecast_type}"]
        else:
            if debias_with == "p":
                wp = 1.0
                wf = 0.0
            elif debias_with == "c":
                wp = 0.0
                wf = 1.0
            else:
                wp = 10. / 11.
                wf = 1. / 11.

            debias_data = wf * load_data[f"cf-{forecast_type}"] + wp * load_data[f"pf-{forecast_type}"]

    """
    Load and merge ground truth 
    """
    printf('Pivoting ground truth to have one row per start_date')
    gt = data_loaders.get_ground_truth(gt_id).loc[:, ['lat', 'lon', 'start_date', gt_col]]

    # Need gt for both the debias and the forecast dates
    gt = gt.loc[gt.start_date.isin(
        debias_data.index.get_level_values("start_date") |
        forecast_data.index.get_level_values("start_date")),
                ['lat', 'lon', 'start_date', gt_col]].set_index(
        ['lat', 'lon', 'start_date']).unstack(['lat', 'lon'])

    printf("Merging ground truth")
    debias_data = debias_data.join(gt, how="left", on="start_date")
    forecast_data = forecast_data.join(gt, how="left", on="start_date")
    # del gt

    # Compute debiasing correction
    # TODO: need to deal with leap years
    # TODO: don't need to compute debiasing for the entire dataframe;
    # should only compute bias for the dates in target dates
    printf('Compute debiasing correction (ground-truth - base prediction) by month-day combination')

    # Compute bias
    bias = (debias_data[gt_col] - debias_data[base_col])

    # Initialize bias per start_date dataframe in forecast dir
    avg_bias = pd.DataFrame(columns=bias.columns,
                            index=bias.index.get_level_values(
                                'model_issuance_date').unique().sort_values())

    for (date, df) in bias.groupby(by="model_issuance_date"):
        # Get all forecasts within +/- 3 days of the day/month of the current forecast
        last_train_date = date - forecast_delta
        debias = bias[
            (bias.index.get_level_values("start_date") >= str(date.year - train_years)) &
            (bias.index.get_level_values("start_date") <= last_train_date) &
            (bias.index.get_level_values("model_issuance_date") <= date + timedelta(days=6)) &
            (bias.index.get_level_values("model_issuance_date") >= date - timedelta(days=6))]

        avg_bias.loc[date] = estimator(debias)

    # Make predictions for each target date
    printf('Creating dataframe to store performance')
    rmses = pd.Series(index=target_date_objs, dtype=np.float64)

    printf('Forming debiased predictions for target dates')
    # Form predictions for target dates in data matrix
    valid_targets = forecast_data.index.intersection(target_date_objs)  # intersect with forecast data
    valid_targets = avg_bias.index.intersection(valid_targets)  # intersect with debiasing data

    preds = forecast_data.loc[valid_targets, base_col] + avg_bias.loc[valid_targets]
    preds.index.name = "start_date"

    # Order valid targets by day of week
    # valid_targets = valid_targets[valid_targets.weekday.argsort(kind='stable')]
    for target_date_obj in valid_targets:
        # Skip if forecast already produced for this target
        target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
        forecast_file = get_forecast_filename(
            model=model_name, submodel=submodel_name,
            gt_id=gt_id, horizon=horizon,
            target_date_str=target_date_str)
        if os.path.isfile(forecast_file) and False:
            printf(f"prior forecast exists for target={target_date_obj}")
            pred = pd.read_hdf(forecast_file).set_index(['lat', 'lon']).pred
        else:
            printf(f'Processing {model_name} forecast for {target_date_obj}')
            # Get prediction
            pred = preds.loc[target_date_obj, :]
            # Save prediction to file in standard format
            if target_date_obj.year >= LAST_SAVE_YEAR:
                save_forecasts(
                    preds.loc[[target_date_obj], :].unstack().rename("pred").reset_index(),
                    model=model_name, submodel=submodel_name,
                    gt_id=gt_id, horizon=horizon,
                    target_date_str=target_date_str)
        # Evaluate and store error if we have ground truth data
        if target_date_obj in forecast_data.index:
            rmse = np.sqrt(np.square(pred - forecast_data.loc[target_date_obj, gt_col]).mean())
            rmses.loc[target_date_obj] = rmse
            printf("-rmse: {}, score: {}".format(rmse, mean_rmse_to_score(rmse)))
            mean_rmse = rmses.mean()
            printf("-mean rmse: {}, running score: {}".format(mean_rmse, mean_rmse_to_score(mean_rmse)))

    printf("Save rmses in standard format")
    rmses = rmses.sort_index().reset_index()
    rmses.columns = ['start_date', 'rmse']
    save_metric(rmses, model=model_name, submodel=submodel_name,
                gt_id=gt_id, horizon=horizon, target_dates=target_dates,
                metric="rmse")
