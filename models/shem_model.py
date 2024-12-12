from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def build_shem_model(X_train, y_train):
    """
    Build and train a Stacked Heterogeneous Ensemble Model (SHEM).
    """
    estimators = [
        ('rf', RandomForestRegressor(random_state=0)),
        ('gbr', GradientBoostingRegressor(random_state=0)),
        ('svr', SVR()),
        ('dt', DecisionTreeRegressor(random_state=0))
    ]
    model = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(random_state=0))
    model.fit(X_train, y_train)
    return model
