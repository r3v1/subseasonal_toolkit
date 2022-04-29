# Constant prediction per month-day-lat-lon combination
#
# Example usage:
#   python src/models/climpp/batch_predict.py contest_tmp2m 34w -t std_val -l rmse -y 26 -m 7
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --loss (-l): cross-grid point loss function used to fit model
#     ("rmse" or "mse"); (default: "rmse")
#   --num_years (-y): number of years to use in training ("all" for all years
#     or positive integer); (default: "all")
#   --margin_in_days (-m): number of month-day combinations on either side of
#     the target combination to include; set to 0 to include only target
#     month-day combo; set to 182 to include entire year; (default: 0)

from argparse import ArgumentParser
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean
from subseasonal_data import data_loaders
from subseasonal_data.utils import get_measurement_variable

from subseasonal_toolkit.utils.eval_util import get_target_dates, mean_rmse_to_score, save_metric
from subseasonal_toolkit.utils.experiments_util import get_id_name, get_th_name, get_start_delta, \
    cond_indices, get_conditioning_cols
from subseasonal_toolkit.utils.general_util import printf
from subseasonal_toolkit.utils.models_util import (get_submodel_name, start_logger, log_params, save_forecasts)


def geometric_median(X: pd.DataFrame, eps=1e-5):
    """
    Yehuda Vardi and Cun-Hui Zhang's algorithm for the geometric median

    Computes the geometric median of the columns of X, up to a tolerance epsilon.

    The geometric median is the vector that minimizes the mean Euclidean norm to
    each column of X.

    References
    ----------
    - https://www.pnas.org/doi/pdf/10.1073/pnas.97.4.1423
    - https://stackoverflow.com/a/30305181
    """
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])  # Compute Euclidean distance between each pair of the two collections of inputs
        nonzeros = (D != 0)[:, 0]  # Comprueba que no haya ceros (luego se va a dividir)

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def ssm(X, alpha=1):
    """Computes stabilized sample mean (Orenstein, 2019) of each column of X

    Args:
        alpha: if infinity, recovers the mean; if 0 approximates median
    """
    # Compute first, second, and third uncentered moments
    mu = np.mean(X, 0)
    mu2 = np.mean(np.square(X), 0)
    mu3 = np.mean(np.power(X, 3), 0)
    # Return mean - (third central moment)/(3*(2+numrows(X))*variance)
    return mu - (mu3 - 3 * mu * mu2 + 2 * np.power(mu, 3)).div(3 * (2 + alpha * X.shape[0]) * (mu2 - np.square(mu)))


if __name__ == "__main__":
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",
                        nargs="*")  # gt_id and horizon
    parser.add_argument('--target_dates', '-t', default="std_test")
    # Loss used to learn parameters
    parser.add_argument('--loss', '-l', default="rmse",
                        help="loss function: mse, rmse, skill, or ssm")
    # Number of years to use in training ("all" or integer)
    parser.add_argument('--num_years', '-y', default="all")
    # Number of month-day combinations on either side of the target combination to include
    # Set to 0 to include only target month-day combo
    # Set to 182 to include entire year
    parser.add_argument('--margin_in_days', '-m', default=0)
    parser.add_argument('--mei', default=False, action='store_true', help="Whether to condition on MEI")
    parser.add_argument('--mjo', default=False, action='store_true', help="Whether to condition on MJO")
    args, opt = parser.parse_known_args()

    # Assign variables
    gt_id = get_id_name(args.pos_vars[
                            0])  # "contest_precip" or "contest_tmp2m"
    horizon = get_th_name(args.pos_vars[1])  # "12w", "34w", or "56w"
    target_dates = args.target_dates
    loss = args.loss
    num_years = args.num_years
    mei = args.mei
    mjo = args.mjo
    if num_years != "all":
        num_years = int(num_years)
    margin_in_days = int(args.margin_in_days)

    #
    # Process model parameters
    #
    # One can subtract this number from a target date to find the last viable training date.
    start_delta = timedelta(days=get_start_delta(horizon, gt_id))

    # Record model and submodel name
    model_name = "climpp"
    submodel_name = get_submodel_name(model_name, loss=loss,
                                      num_years=num_years, margin_in_days=margin_in_days, mei=mei, mjo=mjo)

    FIRST_SAVE_YEAR = 2007  # Don't save forecasts from years prior to than FIRST_SAVE_YEAR

    # Save output to log file
    logger = start_logger(model=model_name, submodel=submodel_name, gt_id=gt_id,
                          horizon=horizon, target_dates=target_dates)
    # Store parameter values in log
    params_names = ['gt_id', 'horizon', 'target_dates',
                    'loss', 'num_years', 'margin_in_days']
    params_values = [eval(param) for param in params_names]
    log_params(params_names, params_values)

    # Select estimator based on loss
    if loss == "rmse":
        estimator = geometric_median
    elif loss == "ssm":
        estimator = ssm
    else:
        estimator = np.mean

    printf('Loading target variable and dropping extraneous columns')
    var = get_measurement_variable(gt_id)
    if loss == "skill":
        clim_col = var + "_clim"
        anom_col = var + "_anom"
        gt = data_loaders.get_ground_truth_anomalies(gt_id).loc[:, ["start_date", "lat", "lon", anom_col]]
        printf('Loading climatology and replacing start date with month-day')
        clim = data_loaders.get_climatology(gt_id)
        clim = clim.set_index(
            ['lat', 'lon', clim.start_date.dt.month, clim.start_date.dt.day]
        ).drop(columns='start_date').squeeze()
    else:
        gt = data_loaders.get_ground_truth(gt_id).loc[:, ["start_date", "lat", "lon", var]]

    printf('Pivoting dataframe to have one column per lat-lon pair and one row per start_date')
    gt = gt.set_index(['lat','lon','start_date']).squeeze().unstack(['lat','lon'])
    if loss == "skill":
        printf('Pivoting climatology to have one column per lat-lon pair and one row per month-day')
        clim = clim.unstack(['lat', 'lon'])
    printf('Dropping any rows with NAs')
    gt = gt.dropna(how='any')
    if loss == "skill":
        printf('L2 normalizing anomaly rows')
        anom = gt.copy()
        gt = gt.div(np.sqrt(np.square(gt).sum(axis=1)), axis=0)

    # Conditioning
    if mei or mjo:
        # conditioning_data = load_combined_data('date_data', gt_id, horizon)
        conditioning_data = data_loaders.load_combined_data('date_data', gt_id, horizon)
        conditioning_columns = get_conditioning_cols(gt_id, horizon, mei=mei, mjo=mjo)
        # Combined data start dates and gt start dates don't fully overlap
        conditioned_targets = pd.DataFrame(gt.index).merge(conditioning_data[["start_date"] + conditioning_columns],
                                                           on="start_date", how="left")

    #
    # Make predictions for each target date
    #
    target_date_objs = pd.Series(get_target_dates(date_str=target_dates, horizon=horizon))
    rmses = pd.Series(index=target_date_objs, dtype=np.float64)
    preds = pd.DataFrame(index=target_date_objs, columns=gt.columns,
                         dtype=np.float64)
    preds.index.name = "start_date"
    # Sort target_date_objs by day of week
    target_date_objs = target_date_objs[target_date_objs.dt.weekday.argsort(kind='stable')]
    X = pd.DataFrame(index=gt.index, columns=["delta", "dividend", "remainder"])
    days_per_year = 365.242199
    for target_date_obj in target_date_objs:
        # tic()
        target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
        # printf(f"Preparing covariates for {target_date_str}")
        # Compute days from target date
        X['delta'] = (target_date_obj - gt.index).days
        # Extract the dividend and remainder when delta is divided by the number of days per year
        # The dividend is analogous to the year
        # (Negative values will ultimately be excluded)
        X['dividend'] = np.floor(X.delta / days_per_year)
        # The remainder is analogous to the day of the year
        X['remainder'] = np.floor(X.delta % days_per_year)
        # Find the last observable training date for this target
        last_train_date = target_date_obj - start_delta
        # Restrict data based on training date, dividend, and remainder
        if mei or mjo:
            target_conditioning_val = \
                conditioning_data[conditioning_data.start_date == target_date_obj][conditioning_columns].values[0]
            indic = cond_indices(conditioned_targets, conditioning_columns, target_conditioning_val)
            indic &= (X.index <= last_train_date)
        else:
            # Coger el conjunto de entrenamiento hasta la última fecha observable para la fecha objetivo
            indic = (X.index <= last_train_date)

        # Filtrar sólo las fechas que coincidan en día y mes (con un margen en días de margin_in_days)
        indic = indic & ((X.remainder <= margin_in_days) | (X.remainder >= 365 - margin_in_days))
        if num_years != "all":
            indic = indic & (X.dividend < num_years)
        if not indic.any():
            printf(f'-Warning: no training data for {target_date_str}; skipping')
            continue

        #################
        # Entrenamiento #
        #################
        printf(f'Fitting climpp model with loss {loss} for {target_date_obj}')
        preds.loc[target_date_obj, :] = estimator(gt.loc[indic, :])

        if loss == "skill":
            # Rescale estimator to minimize MSE over training set
            denom = preds.loc[target_date_obj, :].dot(preds.loc[target_date_obj, :])
            numerator = preds.loc[target_date_obj, :].dot(estimator(anom.loc[indic, :]))
            scale = numerator / (denom + (denom == 0))
            printf(f"-anomaly scale={scale}")
            if numerator < 0:
                # Avoid changing sign: scale down anomaly to be nearly but not exactly zero
                eps = 1e-7
                scale = eps / (np.sqrt(denom) + (denom == 0))
                printf(f"-positive scale={scale}")
            preds.loc[target_date_obj, :] *= scale
            # Add climatology to anomalies
            target_clim = clim.loc[(target_date_obj.month, target_date_obj.day), :]
            preds.loc[target_date_obj, :] += target_clim

        # Save prediction to file in standard format
        if target_date_obj.year >= FIRST_SAVE_YEAR:
            save_forecasts(
                # Vuelve a reconstruir el DataFrame en forma tabular lat-lon-start_date
                preds.loc[[target_date_obj], :].unstack().rename("pred").reset_index(),
                model=model_name, submodel=submodel_name,
                gt_id=gt_id, horizon=horizon,
                target_date_str=target_date_str)
        # Evaluate and store error if we have ground truth data
        if target_date_obj in gt.index:
            if loss == "skill":
                rmse = np.sqrt(
                    np.square(preds.loc[target_date_obj, :] - target_clim - anom.loc[target_date_obj, :]).mean())
            else:
                rmse = np.sqrt(np.square(preds.loc[target_date_obj, :] - gt.loc[target_date_obj, :]).mean())
            rmses.loc[target_date_obj] = rmse
            print("-rmse: {}, score: {}".format(rmse, mean_rmse_to_score(rmse)))
            mean_rmse = rmses.mean()
            print("-mean rmse: {}, running score: {}".format(mean_rmse, mean_rmse_to_score(mean_rmse)))

    printf("Save rmses in standard format")
    rmses = rmses.sort_index().reset_index()
    rmses.columns = ['start_date', 'rmse']
    save_metric(rmses, model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon,
                target_dates=target_dates, metric="rmse")
