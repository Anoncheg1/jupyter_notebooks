import pandas as pd
import os


def read_dfs(files: list) -> pd.DataFrame:
    df_res = None
    for x in files:
        df: pd.DataFrame = pd.read_pickle(x)
        if df_res is None:
            df_res = df
        else:
            df_res = pd.concat([df_res, df], axis=0, sort=False, ignore_index=True)
    return df_res


def load():
    """
    :return: dates, X, Y
    """
    # mypath = '/mnt/hit4/hit4user/PycharmProjects/mysql_connector'
    mypath = '/home/u2/jupyter_notebooks/modules'
    # mypath = 'modules'
    X_files = sorted([os.path.join(mypath, f) for f in os.listdir(mypath) if 'final_features_X' in f])
    Y_files = sorted([os.path.join(mypath, f) for f in os.listdir(mypath) if 'final_features_Y' in f])
    date_files = sorted([os.path.join(mypath, f) for f in os.listdir(mypath) if 'date_' in f])
    X = read_dfs(X_files)
    Y = read_dfs(Y_files)
    dates = read_dfs(date_files)
    X = pd.get_dummies(X, dummy_na=True)  # categorical
    X.fillna(0, inplace=True)  # numerical if NaN > 70%
    Y['under'].replace(2, 0, inplace=True)
    return dates, X, Y
