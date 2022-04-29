# Create batch forecasts using official climatology mean for a specified set of test dates.
#
# Example usage:
#   python src/models/climatology/batch_predict.py contest_tmp2m 34w -t std_test
#
# Positional args:
#   gt_id: contest_tmp2m or contest_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction (default: 'std_test')

from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from cartopy import crs as ccrs
from subseasonal_data.data_loaders import get_climatology

from subseasonal_toolkit.utils.eval_util import get_target_dates
from subseasonal_toolkit.utils.experiments_util import get_id_name, get_th_name
from subseasonal_toolkit.utils.models_util import save_forecasts
import xarray as xr

def visualizar_datos(df: pd.DataFrame, variable: str) -> xr.Dataset:
    # Construímos el dataset
    ds = df.set_index(["start_date", "lat", "lon"]).to_xarray()
    da = ds.to_array().isel(variable=0, start_date=0)

    fig, ax = plt.subplots(
        1,
        1,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(10, 5),
    )

    da.plot(
        ax=ax,
        cmap="RdBu",
        cbar_kwargs=dict(orientation="vertical", fraction=0.0315, pad=0.025),
    )
    ax.coastlines()
    plt.title(f"Extensión de la climatología: {variable}")
    plt.tight_layout()
    plt.show()

    return ds

if __name__ == "__main__":

    # Load command line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon
    parser.add_argument('--target_dates', '-t', default="std_test")
    args = parser.parse_args()

    # Assign variables
    gt_id = get_id_name(args.pos_vars[0])  # "contest_precip" or "contest_tmp2m"
    horizon = get_th_name(args.pos_vars[1])  # "34w" or "56w"
    target_dates = args.target_dates

    model_name = "climatology"
    submodel_name = "climatology"

    official_clim = get_climatology(gt_id, sync=False, allow_write=True)
    # ds = visualizar_datos(official_clim, gt_id)  # Para visualizar la extensión

    official_clim["day"] = official_clim.start_date.dt.day
    official_clim["month"] = official_clim.start_date.dt.month

    target_date_objs = get_target_dates(date_str=target_dates, horizon=horizon)
    for target_date in target_date_objs:
        target_date_str = datetime.strftime(target_date, '%Y%m%d')

        # La climatología ya está calculada, simplemente hay que filtrar según el día y el mes objetivo
        preds = official_clim[(official_clim.day == target_date.day) & (official_clim.month == target_date.month)]
        preds = preds.drop(["day", "month", "start_date"], axis=1)
        preds["start_date"] = target_date
        preds = preds.rename(columns={gt_id.split("_")[1]: 'pred'})

        # Save predictions
        save_forecasts(preds, model=model_name, submodel=submodel_name,
                       gt_id=gt_id, horizon=horizon,
                       target_date_str=target_date_str)
