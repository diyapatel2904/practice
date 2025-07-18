import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline
from functools import partial
from skopt import space
from skopt import gp_minimize
from hyperopt import hp, fmin, tpe, Trials


# if __name__ == "__main__":
#     df= pd.read_csv("/home/diya/practice/Dataset/mobile_price_train.csv")
#     X = df.drop("price_range",axis=1).values
#     y = df.price_range.values

# ---------------------------------------------------
#    
# # GRID SEACHCV

#     param_grid = {
#         "n_estimators":[100,200,300,400],
#         "max_depth":[1,3,5,7],
#         "criterion":["gini","entropy"],
#     }

    # model = model_selection.GridSearchCV(
    #     estimator=classifier,
    #     param_grid=param_grid,
    #     scoring="accuracy",
    #     verbose=10,
    #     cv=5,
    # )
    # model.fit(X,y)
    # print(model.best_score_)
    # print(model.best_estimator_.get_params())

# -----------------------------------------------------------------------------

# # RANDOM SEARCH

    # scl=preprocessing.StandardScaler()
    # pca = decomposition.PCA()
    # rf = ensemble.RandomForestClassifier(n_jobs=-1)          # n_jobs=-1 to use all the course of machine  
    # classifier= pipeline.Pipeline(
    #     [
    #         ("scaling",scl),("pca",pca),("rf",rf)

    #     ]
    # )

    # param_grid = {
    #     "pca__n_components":np.arange(5,10),
    #     "rf__n_estimators": np.arange(100,1500,100),
    #     "rf__max_depth":np.arange(1,20),
    #     "rf__criterion":["gini","entropy"],
    # }


#     classifier = ensemble.RandomForestClassifier(n_jobs=-1)         
#     param_grid = {
#         "n_estimators": np.arange(100,1500,100),
#         "max_depth":np.arange(1,20),
#         "criterion":["gini","entropy"],
    # }


    # model = model_selection.RandomizedSearchCV(
    #     estimator=classifier,
    #     param_distributions=param_grid,
    #     n_iter=10,
    #     scoring="accuracy",
    #     verbose=10,
    #     cv=5,
    # )
    # model.fit(X,y)
    # print(model.best_score_)
    # print(model.best_estimator_.get_params())

# ----------------------------------------------------------------------


# def optimize(params,param_names,x,y):
#     raw_params = dict(zip(param_names,params))
#     param_distributions = {k: [v] for k, v in raw_params.items()}
#     classifier = ensemble.RandomForestClassifier(n_jobs=-1)  

#     model = model_selection.RandomizedSearchCV(
#         estimator=classifier,
#         param_distributions=param_distributions,
#         n_iter=1,  # Only 1 because we are passing fixed values each time via gp_minimize
#         cv=5,
#         scoring="accuracy",
#         n_jobs=-1,
#         verbose=0
#     )

#     kf = model_selection.StratifiedKFold(n_splits=5)
#     accuracies=[]

#     for train_idx, test_idx in kf.split(X=x, y=y):
#         xtrain = x[train_idx]
#         ytrain = y[train_idx]

#         xtest = x[test_idx]
#         ytest = y[test_idx]

#         model.fit(xtrain, ytrain)
#         preds = model.predict(xtest)
#         fold_acc = metrics.accuracy_score(ytest, preds)
#         accuracies.append(fold_acc)

#     return -1.0*np.mean(accuracies)

# if __name__ == "__main__":
#     df= pd.read_csv("/home/diya/practice/Dataset/mobile_price_train.csv")
#     X = df.drop("price_range",axis=1).values
#     y = df.price_range.values

#     param_space = [

#         space.Integer(3,15,name="max_depth"),
#         space.Integer(100,600,name="n_estimators"),
#         space.Categorical(["gini","entropy"],name="criterion"),
#         space.Real(0.01,1,prior="uniform",name="max_features")
#     ]

#     param_names = [ "max_depth", "n_estimators", "criterion", "max_features" ]

#     optimization_function = partial(optimize,param_names=param_names,x=X,y=y)

#     result = gp_minimize (
#         optimization_function,
#         dimensions=param_space,
#         n_calls=15,
#         n_random_starts=10,
#         verbose=10,

#     )


# print(dict(zip(param_names, result.x)))

# ------------------------------------------------------------------------


def optimize(params,param_names,x,y):
    
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)  

    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        n_iter=1,  # Only 1 because we are passing fixed values each time via gp_minimize
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0
    )

    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies=[]

    for train_idx, test_idx in kf.split(X=x, y=y):
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)

    return -1.0*np.mean(accuracies)

if __name__ == "__main__":
    df= pd.read_csv("/home/diya/practice/Dataset/mobile_price_train.csv")
    X = df.drop("price_range",axis=1).values
    y = df.price_range.values

    param_space = [

        "max_depth":hp.quniform("max_depth",3,15,1),
        "n_estimators":hp.quniform("n_estimators",100,600,1),
        "criterion": hp.choice("criterion","gini","entropy"),
        "max_features": hp.quniform("max_features",0.01,1)
    ]

    param_names = [ "max_depth", "n_estimators", "criterion", "max_features" ]

    optimization_function = partial(optimize, x=X, y=y)

    result = gp_minimize (
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10,

    )


print(dict(zip(param_names, result.x)))

