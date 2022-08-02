import pandas as pd
import numpy as np

def get1(p: str):
    df: pd.DataFrame = pd.read_pickle(p)
    # 1)
    # df1 = df.loc[(df['СФ андерайтера'] == 0) & (df['Коды отказа'] != 97) & (df['Коды отказа'] != 91)]
    df.loc[(df['СФ андерайтера'] == 0) & (df['Коды отказа'] != 97) & (df['Коды отказа'] != 91), "target"] = True
    df['target'].fillna(False, inplace=True)
    df.drop('СФ андерайтера', axis=1, inplace=True)
    df.drop('Коды отказа', axis=1, inplace=True)
    df.drop('СФ системы', axis=1, inplace=True)
    df.drop('Статус заявки', axis=1, inplace=True)
    return df


def get2(p: str):
    df: pd.DataFrame = pd.read_pickle(p)
    # 2)
    # df2 = df.loc[(df['Коды отказа'] == 91) | (df['Коды отказа'] == 97)]  # 4012 in df3
    df.loc[(df['Коды отказа'] == 91) | (df['Коды отказа'] == 97), "target"] = True
    df['target'].fillna(False, inplace=True)
    df.drop('СФ андерайтера', axis=1, inplace=True)
    df.drop('Коды отказа', axis=1, inplace=True)
    df.drop('СФ системы', axis=1, inplace=True)
    df.drop('Статус заявки', axis=1, inplace=True)
    return df


def get12(p: str):
    df: pd.DataFrame = pd.read_pickle(p)
    # 2)
    # df2 = df.loc[(df['Коды отказа'] == 91) | (df['Коды отказа'] == 97)]  # 4012 in df3
    df.loc[(df['Коды отказа'] == 91) | (df['Коды отказа'] == 97), "target"] = True
    df['target'].fillna(False, inplace=True)
    df.drop('СФ андерайтера', axis=1, inplace=True)
    df.drop('Коды отказа', axis=1, inplace=True)
    df.drop('СФ системы', axis=1, inplace=True)
    df.drop('Статус заявки', axis=1, inplace=True)
    return df

def get3(p: str):
    df: pd.DataFrame = pd.read_pickle(p)
    # print(df.shape)
    # print(df.loc[df['Статус заявки'] == 'Заявка отменена'])
    # # print(df['Статус заявки'])
    # return
    # 3)
    # df3 = df.loc[(df['Статус заявки'] == 1) & (df['Коды отказа'] != 91) & (df['Коды отказа'] != 97)]
    df.loc[(df['Статус заявки'] == 1) & (df['Коды отказа'] != 91) & (df['Коды отказа'] != 97), "target"] = True
    df['target'].fillna(False, inplace=True)
    # print(df.loc[df['Статус заявки'] == 'Заявка отменена'])
    df.drop('СФ андерайтера', axis=1, inplace=True)
    df.drop('Коды отказа', axis=1, inplace=True)
    df.drop('СФ системы', axis=1, inplace=True)
    df.drop('Статус заявки', axis=1, inplace=True)

    return df


def feature_importance_forest(df: pd.DataFrame, max_depth=12, n_estimators=25, max_leaf_nodes=14):
    """
    :param df:
    :param max_depth:
    :param n_estimators:
    :param max_leaf_nodes:+5
    """
    # df: pd.DataFrame = pd.read_pickle(p)

    # матрица корреляции
    # переставляем столбцы
    # cols = df.columns.to_list()
    # cols = cols[-1:] + cols[:-1]
    # df = df[cols]

    # import seaborn
    # import matplotlib.pyplot as plt
    #
    # print(df.columns.values)
    # seaborn.heatmap(df.corr(), annot=True, )
    # plt.subplots_adjust(right=1)
    # plt.show()

    X = df.drop(['target'], 1)
    Y = df['target']

    from sklearn.ensemble import RandomForestClassifier

    importance_sum = np.zeros(X.shape[1], dtype=np.float)
    n = 100
    max_depth = np.linspace(2, max_depth+8, 100)  # 12
    n_estimators = np.linspace(5, n_estimators+15, 100)  # 25
    max_leaf_nodes = np.linspace(max_leaf_nodes-4, max_leaf_nodes+8, 100)  # 14
    min_samples_split = 2

    for i in range(n):
        depth = int(round(max_depth[i]))
        n_est = int(round(n_estimators[i]))
        max_l = int(round(max_leaf_nodes[i]))

        model = RandomForestClassifier(random_state=i, max_depth=depth,
                                       n_estimators=n_est, max_leaf_nodes=max_l,
                                       min_samples_split=2)
        model.fit(X, Y)
        # FEATURE IMPORTANCE
        importances = model.feature_importances_  # feature importance
        importance_sum += importances

    indices = np.argsort(importance_sum)[::-1]  # sort indexes

    # Print the feature ranking
    print("Feature ranking:")
    print(importance_sum.shape)

    for f in range(X.shape[1])[:10]:  # первые 10
        print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importance_sum[indices[f]] / 10))


def feature_importance_xgboost(df):

    # df: pd.DataFrame = pd.read_pickle('feature_eng.pickle')
    from xgboost import XGBClassifier
    X = df.drop(['target'], 1)
    Y = df['target']
    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, eval_metric='auc',
              gamma=1.2, gpu_id=-1, importance_type='gain',
              interaction_constraints='', learning_rate=0.2, max_delta_step=0,
              max_depth=7, min_child_weight=1,
              monotone_constraints='()', n_estimators=1, n_jobs=4,
              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1, tree_method='exact',
              use_label_encoder=False, validate_parameters=1, verbosity=None)
    # model = XGBClassifier(base_score=0.5, booster='gbtree', eval_metric='auc')
    # model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #           colsample_bynode=1, colsample_bytree=1,
    #                       gamma=0, gpu_id=-1, importance_type='gain',
    #                       interaction_constraints='', learning_rate=0, max_delta_step=0,
    #                       max_depth=3, min_child_weight=1, missing=np.nan, monotone_constraints='()',
    #           n_estimators=1, n_jobs=4, num_parallel_tree=1, random_state=0,
    #           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
    #           tree_method='exact', validate_parameters=1, verbosity=None)
    model = model.fit(X, Y)
    # Z = model.predict(X)
    # print(sum(Z))

    # print(Y.unique())
    # print(model.score(X, Y))
    # print(model)
    # print(Y[Y == False])
    # print(model.predict(X.iloc[[36107]]))
    # print(model.predict(X.iloc[[36111]]))
    # print(model.predict(X.iloc[[36115]]))
    # print(model.predict(X.iloc[[36141]]))
    # print(model.predict(X.iloc[[36142]]))
    # print(model.predict(X.iloc[[36143]]))

    # print(model.score(X, Y))
    # print(model.feature_importances_)  # ?
    from xgboost import Booster
    booster: Booster = model.get_booster()
    print(booster.get_dump()[0])
    # return
    # --- PLOT FIRST TREE
    # from xgboost import plot_tree
    # import matplotlib.pyplot as plt
    # plot_tree(model)
    # plt.show()
    # --- PLOT FEATURE IMPORTANCE
    from xgboost import plot_importance
    import matplotlib.pyplot as plt
    plot_importance(model)
    plt.show()

