from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sklearn
import xgboost as xgb
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import datetime

from modules.rf_importance import calc_feature_importance


date_pick_b = widgets.DatePicker(
    description='Начало выборки',
    disabled=False,
    style={'description_width': 'initial'}
)
date_pick_b.value = datetime.date(year=2019, month=11, day=20)

date_pick_f = widgets.DatePicker(
    description='Конец выборки',
    disabled=False,
    style={'description_width': 'initial'}
)


def f(d1, d2, dates, X, Y):
    dates = dates.reset_index(drop=True)
    if d1 and d2:
        dt1 = datetime.datetime(d1.year, d1.month, d1.day)
        dt2 = datetime.datetime(d2.year, d2.month, d2.day)
        index1 = dates[dt1 <= dates].index
        index2 = dates[dates <= dt2].index
        if len(index1) == 0 or len(index2) == 0:
            print("No data")
            return
        intersection = sorted(set(index2.to_list()).intersection(set(index1.to_list())))

        X_short = X.iloc[intersection]
        Y_short = Y.iloc[intersection]

        # indices, importa`nce_sum = calc_feature_importance(X_short, Y_short, n=150)
        # Print the feature ranking
        print("Feature ranking:")
        for p in range(X_short.shape[1])[:100]:  # первые 50
            print("%d. %s (%f)" % (p + 1, X_short.columns[indices[p]], importance_sum[indices[p]] / 100))


if __name__ == '__main__':
    from modules.load import load

    dates, X, Y = load()
    f(datetime.date(year=2019, month=11, day=20),
      datetime.date(year=2020, month=11, day=20), dates, X, Y)
