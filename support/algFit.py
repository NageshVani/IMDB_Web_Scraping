# Make the model and perform MAPE Error
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
import pandas as pd

def algorthm_test(algorthm, dftrain, dftest, predictors, target):
    y_pred=[]
    train_pred=[]
    #Fit the algorthmorithm on the data
    algorthm.fit(dftrain[predictors], dftrain[target]) 
    
    #Predict training set:
    dftrain_predictions = algorthm.predict(dftrain[predictors])
    
    #Perform cross-validation:
    cv_score = cross_val_score(algorthm, dftrain[predictors], dftrain[target], cv=20, n_jobs=-1, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    # RMSE value
    rm_se = np.sqrt(metrics.mean_squared_error(dftrain[target].values, dftrain_predictions))
    
    #Predict on testing data:
    y_pred1 = algorthm.predict(dftest[predictors])
    for i in range(0,len(y_pred1)):
        conv = "%.1f" % y_pred1[i]
        y_pred.append(float(conv))
        
    # check MAPE error
    y_true = dftest[target]
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    err = np.mean(np.abs((y_true - y_pred) / y_true))
    
    #Print model report:
    print ("\n------Model Report----")
    print ("RMSE : %s" % "{0:.3%}" .format(rm_se) )
    print('MAPE Error: %s' % "{0:.3%}" .format(err))
    print ("CV Score Mean : %.4g" %(np.mean(cv_score)))
    print ("CV Score Std : %.4g" %(np.std(cv_score)))
        
    # print sample values
    print ("\n----------Sample values---")
    print('y_true values', y_true[0:8])
    print('y_pred values', y_pred[0:8])
    