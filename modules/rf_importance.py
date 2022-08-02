import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

def calc_feature_importance(X_short, Y_short, n: int=100):
    """

    :param X_short:
    :param Y_short:
    :param n: loops
    :return:
    """
    importance_sum = np.zeros(X_short.shape[1], dtype=np.float)
    # ВАЖНОСТЬ ПАРАМЕТРОВ
    max_depth = np.linspace(7, 20, n)  # 12
    n_estimators = np.linspace(5, 40, n)  # 25
    max_leaf_nodes = np.linspace(8, 20, n)  # 14
    min_samples_split = 2

    if n > 100:
        rangemy = tqdm(range(n))
    else:
        rangemy = range(n)

    for i in rangemy:
        depth = int(round(max_depth[i]))
        n_est = int(round(n_estimators[i]))
        max_l = int(round(max_leaf_nodes[i]))

        model = RandomForestClassifier(random_state=i, max_depth=depth,
                                       n_estimators=n_est, max_leaf_nodes=max_l,
                                       min_samples_split=min_samples_split)
        model.fit(X_short, Y_short['under'])
        # FEATURE IMPORTANCE
        importances = model.feature_importances_  # feature importance
        importance_sum += importances

    indices = np.argsort(importance_sum)[::-1]  # sort indexes
    return indices, importance_sum
