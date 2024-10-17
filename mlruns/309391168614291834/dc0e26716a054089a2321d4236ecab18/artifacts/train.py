import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,precision_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1] 


rf = RandomForestClassifier(random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    
    'n_estimators' : [10, 20, 30, 100],
    'max_depth' : [None, 5, 10 , 20, 30]
}
    
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv = 5, n_jobs=-1, verbose=2)




# creating new exp to be tracked
mlflow.set_experiment('diabetes-rf-hp')

with mlflow.start_run(description="Best hyperparameter trained RF model",run_name='grid-search-rf') as parent: # in parent we log the best model
    
    grid_search.fit(X_train,y_train)
    
    
    # log all the children runs
    for i in range(len(grid_search.cv_results_['params'])):
        # for every combination will run a with statement:
        with mlflow.start_run(nested=True) as child: # nested = True means child runs will happen
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric('accuracy', grid_search.cv_results_['mean_test_score'][i])


    # Displaying best params and best score

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_  
    
    # logging:
    # params
    mlflow.log_params(param_grid)

    # metric
    mlflow.log_metric('accuracy', best_score)
    
    # data
    
# data
    train_df = X_train.copy()
    train_df['Outcome'] = y_train

    train_df = mlflow.data.from_pandas(train_df)

    mlflow.log_input(train_df, "training")

    test_df = X_test.copy()
    test_df['Outcome'] = y_test

    test_df = mlflow.data.from_pandas(test_df)

    mlflow.log_input(test_df, "training")
    
    # source code:
    mlflow.log_artifact(__file__)
    
    
    print(X_train.head())
    
    # signature

    signature = mlflow.models.infer_signature(X_train, grid_search.best_estimator_.predict(X_train))
        
    # model:
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_random_forest", signature=signature)
    
     # mlflow tags
    mlflow.set_tags({'author':'Mohit','model':'random_forest'})
    
    
     
    print('best_params',best_params)

    print('best_score',best_score)  
    
    



