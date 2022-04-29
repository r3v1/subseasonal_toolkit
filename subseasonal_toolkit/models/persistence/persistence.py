from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from subseasonal_data import data_loaders
from subseasonal_data.utils import get_measurement_variable

from subseasonal_toolkit.utils.eval_util import get_target_dates, mean_rmse_to_score, save_metric
from subseasonal_toolkit.utils.experiments_util import get_start_delta
from subseasonal_toolkit.utils.general_util import printf
from subseasonal_toolkit.utils.models_util import (get_submodel_name, save_forecasts)

if __name__ == "__main__":
    #
    # Specify model parameters
    #

    # Otherwise, specify arguments interactively
    gt_id = "us_precip"
    horizon = "34w"
    target_dates = "std_paper"

    #
    # Process model parameters
    #
    # One can subtract this number from a target date to find the last viable training date.
    start_delta = timedelta(days=get_start_delta(horizon, gt_id))

    # Record model and submodel name
    model_name = "persistence"
    submodel_name = get_submodel_name(model_name)

    FIRST_SAVE_YEAR = 2007  # Don't save forecasts from years prior to FIRST_SAVE_YEAR

    printf('Loading target variable and dropping extraneous columns')
    var = get_measurement_variable(gt_id)
    gt = data_loaders.get_ground_truth(gt_id).loc[:, ["start_date", "lat", "lon", var]]

    printf('Pivoting dataframe to have one column per lat-lon pair and one row per start_date')
    gt = gt.set_index(['lat', 'lon', 'start_date']).squeeze().unstack(['lat', 'lon'])

    target_date_objs = pd.Series(get_target_dates(date_str=target_dates, horizon=horizon))
    rmses = pd.Series(index=target_date_objs, dtype=np.float64)
    preds = pd.DataFrame(index=target_date_objs, columns=gt.columns,
                         dtype=np.float64)
    preds.index.name = "start_date"
    # Sort target_date_objs by day of week
    target_date_objs = target_date_objs[target_date_objs.dt.weekday.argsort(kind='stable')]
    for target_date_obj in target_date_objs:
        target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
        # Find the last observable training date for this target
        last_train_date = target_date_obj - start_delta
        if not last_train_date in gt.index:
            printf(f'-Warning: no persistence prediction for {target_date_str}; skipping')
            continue
        printf(f'Forming persistence prediction for {target_date_obj}')
        preds.loc[target_date_obj, :] = gt.loc[last_train_date, :]
        # Save prediction to file in standard format
        if target_date_obj.year >= FIRST_SAVE_YEAR:
            save_forecasts(
                preds.loc[[target_date_obj], :].unstack().rename("pred").reset_index(),
                model=model_name, submodel=submodel_name,
                gt_id=gt_id, horizon=horizon,
                target_date_str=target_date_str)
        # Evaluate and store error if we have ground truth data
        if target_date_obj in gt.index:
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
