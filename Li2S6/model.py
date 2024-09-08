from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from xgboost.sklearn import XGBRegressor
from sklearn.inspection import permutation_importance
import joblib


X = np.loadtxt("S6-feature.txt")
Y = np.loadtxt("S6-energy.txt")
kfold = KFold(n_splits=5, shuffle=True)

score = []
def main():
    for i in range(1, 2):
        fold = 1
        for train, test in kfold.split(X, Y):
            reg_1 = KNeighborsRegressor(n_neighbors=2)
            reg_2 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, subsample=0.9, max_depth=5)
            reg_3 = XGBRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8, max_depth=5)
            reg = VotingRegressor(estimators=[('KNN', reg_1), ('GBDT', reg_2), ('XGBoost', reg_3)], weights = [3, 2, 4])
            reg.fit(X[train], Y[train])

            # save models
            joblib.dump(reg, "S6-model/"+str(fold)+".model")

            # get the feature S6-importance by permutation S6-importance method
            result = permutation_importance(reg, X[train], Y[train], n_repeats=5, random_state=42)
            np.savetxt("S6-importance/"+str(fold)+"_importance.txt", result.importances)
            fold = fold + 1

            # calculate the mae
            mae = mean_absolute_error(Y[test], reg.predict(X[test]))
            score.append(mae)
            #mse = mean_squared_error(Y[test], reg.predict(X[test]))
            #accuracy = accuracy_score(Y[test], eclf.predict(X[test]))
            print(Y[test].T, reg.predict(X[test]).T)
            #print(mae)


if __name__ == '__main__':
    main()

