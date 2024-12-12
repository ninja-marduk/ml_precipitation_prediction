from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def select_features(X, y):
    """
    Perform Recursive Feature Elimination with Cross-Validation (RFECV).
    """
    estimator = RandomForestRegressor(random_state=0)
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X, y)

    # Actualiza esta línea para versiones recientes de scikit-learn
    plt.figure()
    plt.title("Feature selection with RFECV")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross-validation score")
    plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), selector.cv_results_['mean_test_score'])
    plt.show()

    return X.columns[selector.support_]
