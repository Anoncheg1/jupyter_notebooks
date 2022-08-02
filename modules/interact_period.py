import ipywidgets as widgets
import pandas as pd
import datetime
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm
# own
from modules.rf_importance import calc_feature_importance

w_days = widgets.IntSlider(
    value=0,
    min=0,
    max=100,
    step=1,
    description='Интервал Дней:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    style={'description_width': 'initial'}
)

w_months = widgets.IntSlider(
    value=0,
    min=0,
    max=6,
    step=1,
    description='Или Месяцов:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    style={'description_width': 'initial'}
)

w_rec = widgets.IntSlider(
    value=2000,
    min=100,
    max=7000,
    step=100,
    description='Строк в интервале:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    style={'description_width': 'initial'}
)


def to_printable(fi_res: OrderedDict) -> dict:
    """
    :param fi_res: {dt:[cols]}
    :return: unic_cols_pos {cols:[-ranks]}
    """
    unic_cols = [col for k in fi_res for col in fi_res[k]]
    unic_cols_pos = {c: [None for _ in range(len(fi_res))] for c in unic_cols}
    # print(unic_cols_pos)

    for col in unic_cols:
        for i, k in enumerate(fi_res):
            for rank, c in enumerate(fi_res[k]):
                if c == col:
                    unic_cols_pos[col][i] = -rank

    return unic_cols_pos


def plot_features_intime(unic_cols_pos: dict, date_points: list):
    """ Sort and print"""

    x = [x.to_pydatetime() for x in date_points]  # hack for matplotlib

    plt.figure(1)
    ax = plt.gca()

    for c, ranks in unic_cols_pos.items():
        # print(ranks)
        plt.plot(x, ranks, label=c)
    plt.figure(2)
    plt.plot([0], [None])  # hack for jupyter
    plt.figlegend(*ax.get_legend_handles_labels(), loc='lower left')
    plt.show()


def filter_top_cols(fi_res: OrderedDict):
    """  10 top without None
    :param fi_res:
    :return: unic_cols_pos
    """
    # filter with None
    unic_cols_pos: dict = to_printable(fi_res)  # Call

    for col, ranks in list(unic_cols_pos.items()):
        if None in ranks:
            del unic_cols_pos[col]

    # filter 10 most sum rank
    cols_top_rank = [(col, sum(ranks)) for col, ranks in unic_cols_pos.items()]
    cols_top_rank = sorted(cols_top_rank, key=lambda x: x[1], reverse=True)[:10]
    top_cols = [x[0] for x in cols_top_rank]

    unic_cols_pos = {k: v for k, v in unic_cols_pos.items() if k in top_cols}
    return unic_cols_pos


def f(rec: int, days: int, months: int, dates: pd.DataFrame, X: pd.DataFrame, Y: pd.DataFrame):
    if rec == 0:
        return
    elif days and months:
        print("select only one")
    elif days or months:
        fi_res = OrderedDict()
        l = len(dates)
        final_dt = dates.iloc[-1]
        new_dt: pd.Timestamp = final_dt
        indexes_newdt = []
        indexes = dates[dates < new_dt].index
        indexes_newdt.append((indexes, new_dt))
        # records < date
        while True:
            if days:
                new_dt = new_dt - datetime.timedelta(days=days)
            else:
                for _ in range(months):
                    dt: datetime.datetime = new_dt.to_pydatetime()
                    new_dt = new_dt - datetime.timedelta(days=dt.day)

            indexes = dates[dates < new_dt].index  # records < date
            if len(indexes) == 0:
                break
            indexes_newdt.append((indexes, new_dt))

        for x in tqdm(indexes_newdt):
            indexes, new_dt = x
            # print(len(indexes), new_dt)
            ind = indexes[-1]  # last index
            # get record from ind to -rec
            diff = l - ind
            X_short = X.iloc[-rec - diff:ind, :]
            Y_short = Y.iloc[-rec - diff:ind, :]
            indices, importance_sum = calc_feature_importance(X_short, Y_short)  # Call

            fi_res[new_dt] = []
            for p in range(X_short.shape[1])[:100]:  # первые 100
                fi_res[new_dt].append(str(X_short.columns[indices[p]]))

        # with open("a.pkl", "wb") as f:
        #     pickle.dump(fi_res, f)
        # exit(0)
        print("Пять самых значимых характеристик:")
        fi_five_res = {k: v[:5] for k, v in fi_res.items()}
        fi_five_res = OrderedDict(sorted(fi_five_res.items()))  # sort asc by date
        unic_cols_pos = to_printable(fi_five_res)  # Call
        plot_features_intime(unic_cols_pos, list(fi_five_res.keys()))  # Call

        print("Десять самых стабильных и значимых характеристик:")
        unic_cols_pos = filter_top_cols(fi_res)
        plot_features_intime(unic_cols_pos, list(fi_res.keys()))


if __name__ == '__main__':
    from modules.load import load
    dates, X, Y = load()
    f(1000, 0, 1, dates, X, Y)
    exit(0)
    with open("a.pkl", "rb") as f:
        fi_res: OrderedDict = pickle.load(f)

    unic_cols_pos = filter_top_cols(fi_res)
    print(fi_res.keys())
    plot_features_intime(unic_cols_pos, list(fi_res.keys()))

    exit(0)
    #
    # # a = np.array([[0.3,0.4], [0.4, 0.2]])
    # # x = [datetime.datetime(year=2020, day=x, month=1) for x in range(1, 6)]
    #
    x = [0, 1, 2, 3, 4]
    y1 = [2, 3, 5, 7, 5]
    y2 = [2, 3, 7, 7, 8]

    from matplotlib import pylab

    plt.figure()
    ax = plt.gca()
    plt.plot(x, y1, label="1")
    plt.plot(x, y2, label="2")

    plt.figure()
    plt.figlegend(*ax.get_legend_handles_labels(), loc='upper left')
    plt.show()
    # a = plt.plot(x, y1, label="1")
    # b = plt.plot(x, y2, label="2")
    # # plt.legend(loc="upper right")
    # plt.legend()
    # # b = plt.plot([1, 2, 3], [1, 4, 5])
    # plt.show()
