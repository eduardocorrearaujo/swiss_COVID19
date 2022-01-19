#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 14:37:51 2022

@author: eduardoaraujo
"""

import pandas as pd 
import numpy as np 
import copy
#import optuna  # pip install optuna
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import KFold
#from optuna.integration import LightGBMPruningCallback
#import seaborn as sns
from ngboost import NGBRegressor
from ngboost.learners import default_tree_learner
from ngboost.distns import  LogNormal
from ngboost.scores import LogScore
import time
#import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates
fmt = mdates.DateFormatter('%d-%b\n%Y') 
from datetime import datetime, timedelta
#from sklearn.metrics import mean_squared_error, mean_absolute_error, plot_confusion_matrix
#from get_data import *
from sqlalchemy import create_engine
from scipy.signal import correlate, correlation_lags
import scipy.cluster.hierarchy as hcluster
import matplotlib.pyplot as plt 
#import streamlit as st 

engine_public = create_engine("postgresql://epigraph:epigraph@localhost:5432/epigraphhub")
engine_private = create_engine("postgresql://epigraph:epigraph@localhost:5432/privatehub")

def build_lagged_features(dt, maxlag=2, dropna=True):
    '''
    Builds a new DataFrame to facilitate regressing over all possible lagged features
    :param dt: Dataframe containing features
    :param maxlag: maximum lags to compute
    :param dropna: if true the initial rows containing NANs due to lagging will be dropped
    :return: Dataframe
    '''
    if type(dt) is pd.DataFrame:
        new_dict = {}
        for col_name in dt:
            new_dict[col_name] = dt[col_name]
            # create lagged Series
            for l in range(1, maxlag + 1):
                new_dict['%s_lag%d' % (col_name, l)] = dt[col_name].shift(l)
        res = pd.DataFrame(new_dict, index=dt.index)

    elif type(dt) is pd.Series:
        the_range = range(maxlag + 1)
        res = pd.concat([dt.shift(i) for i in the_range], axis=1)
        res.columns = ['lag_%d' % i for i in the_range]
    else:
        print('Only works for DataFrame or Series')
        return None
    if dropna:
        return res.dropna()
    else:
        return res
#@st.cache   
def get_lag(x,y,maxlags=5, smooth=True):
    if smooth:
        x = pd.Series(x).rolling(7).mean().dropna().values
        y = pd.Series(y).rolling(7).mean().dropna().values
    corr = correlate(x,y, mode='full')/np.sqrt(np.dot(x,x)*np.dot(y,y))
    slice = np.s_[(len(corr)-maxlags)//2:-(len(corr)-maxlags)//2]
    corr = corr[slice]
    lags = correlation_lags(x.size,y.size,mode='full')
    lags = lags[slice]
    lag = lags[np.argmax(corr)]

#     lag = np.argmax(corr)-(len(corr)//2)

    return lag, corr.max()

#@st.cache  
def lag_ccf(a,maxlags=30,smooth=True):
    """
    Calculate the full correlation matrix based on the maximum correlation lag 
    """
    ncols = a.shape[1]
    lags = np.zeros((ncols,ncols))
    cmat = np.zeros((ncols,ncols))
    for i in range(ncols): 
        for j in range(ncols):
#             if j>i:
#                 continue
            lag, corr = get_lag(a.T[i],a.T[j], maxlags, smooth)
            cmat[i,j] = corr
            lags[i,j] = lag
    return cmat,lags

#@st.cache  
def compute_clusters(curve, t, plot = False):
    '''
    Function to compute the clusters 
    
    param curve: string. Represent the curve that will used to cluster the regions.
    
    param t: float. Represent the value used to compute the distance between the cluster and so 
    decide the number of clusters 
    
    return: array. 
    -> cluster: is the array with the computed clusters
    -> all_regions: is the array with all the regions
    '''
    
    df = pd.read_sql_table(f'foph_{curve}', engine_public, schema = 'switzerland', columns = ['datum','geoRegion',  'entries'])
    df.index = pd.to_datetime(df.datum)
    
    inc_canton = df.pivot(columns='geoRegion', values='entries')
    
    #changin the data
    #print(inc_canton)
    
    # Computing the correlation matrix based on the maximum correlation lag 
    
    del inc_canton['CHFL']
    
    del inc_canton['CH']
    
    del inc_canton['FL']
    
    cm,lm=lag_ccf(inc_canton.values)
    
    # Plotting the dendrogram
    linkage = hcluster.linkage(cm, method='complete')
   
   
    if plot: 
        fig, ax = plt.subplots(1,1, figsize=(15,10), dpi = 300)
        hcluster.dendrogram(linkage, labels=inc_canton.columns, color_threshold=0.3, ax=ax)
        ax.set_title('Result of the hierarchical clustering of the series', fontdict= {'fontsize': 20})
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        
    else: 
        fig = None
    

    # computing the cluster 
    
    ind = hcluster.fcluster(linkage, t, 'distance')

    grouped = pd.DataFrame(list(zip(ind,inc_canton.columns))).groupby(0) # preciso entender melhor essa linha do código 
    clusters = [group[1][1].values for group in grouped]
    
    all_regions = df.geoRegion.unique() 
    
    return clusters, all_regions, fig

def get_updated_data(smooth):
    
    '''
    Function to get the updated data for Geneva
    
    param smooth: Boolean. If True, a rolling average is applied
    
    return: dataframe. 
    '''
    
    df = pd.read_sql_table('hug_hosp_data', engine_private, schema = 'switzerland', columns = ['Date_Entry', 'Patient_id'])
   
    df.index = pd.to_datetime(df.Date_Entry)
    df_hosp = df.resample('D').count()
    df_hosp = df_hosp[['Patient_id']]
    
    if smooth == True:
        df_hosp  = df_hosp[['Patient_id']].rolling(window = 7).mean()
        df_hosp = df_hosp.dropna()
        
    df_hosp = df_hosp.sort_index()
    df_hosp.rename(columns= {'Patient_id':'hosp_GE'}, inplace = True)
    return df_hosp['2021-09-01':]
    

#@st.cache  
def get_cluster_data(curve, georegion):
    
    '''
    This function provide a dataframe with the curve selected in the param curve for each region selected in the 
    param georegion
    
    param curve: string. The following options are accepted: ['cases', 'death',
                                                              'hosp', 'hospCapacity', 
                                                              're', 'test', 'testPcrAntigen', 'virusVariantsWgs']
    param georegion: array with all the geoRegions of interest.
    
    return dataframe
    '''

        
    df = get_canton_data(curve, georegion)
    #print(df)
    # dataframe where will the curve for each region
    
    df_end = pd.DataFrame()
    
    for i in georegion:
        
        #print(curve)
        #print(i)
        
        if curve == 'hospcapacity':
            df_aux = df.loc[df.geoRegion == i].resample('D').mean()   
            df_end['ICU_patients_'+i] = df_aux.ICU_Covid19Patients
            df_end['total_hosp_' + i] = df_aux.Total_Covid19Patients
            df_end.index = pd.to_datetime(df_end.index)
            df_end[f'diff_total_hosp_{i}'] = np.concatenate( ([np.nan], np.diff(df_aux.Total_Covid19Patients,1)) )
                
        
        else:
            if curve == 'hosp':
                df_aux = df.loc[df.geoRegion == i].resample('D').mean()   
                #df_end['total_hosp_'+i] = df.loc[df.geoRegion == i].sumTotal
                df_end[curve+'_'+i] = df_aux.entries
                
                df_end[f'diff_{curve}_{i}'] = np.concatenate( ([np.nan], np.diff(df_aux.entries,1)))
             
            else:
                df_aux = df.loc[df.geoRegion == i].resample('D').mean()   
                df_end[curve+'_'+i] = df_aux.entries
                df_end[f'diff_{curve}_{i}'] = np.concatenate( ([np.nan], np.diff(df_aux.entries,1)))
             
    df_end = df_end.resample('D').mean()   
        
    return df_end


#@st.cache  
def get_combined_data( data_types, georegion,vaccine = True, smooth = True):
    
    '''
    This function provide a dataframe with the all the curves selected in the param data_types for each region selected in the 
    param georegion
    
    param data_types: array. The following options are accepted: ['cases', 'casesVaccPersons', 'covidCertificates', 'death',
                                                             'deathVaccPersons', 'hosp', 'hospCapacity', 'hospVaccPersons',
                                                             'intCases', 're', 'test', 'testPcrAntigen', 'virusVariantsWgs']
    param georegion: array with all the geoRegions of interest.
    
    return dataframe
    '''

    for i in np.arange(0, len(data_types)):
        
        if i == 0:
            
            df = get_cluster_data(data_types[i], georegion)
            
        else:
            
            df = df.merge(get_cluster_data(data_types[i], georegion), left_index =True, right_index = True)
    
    if vaccine == True:
        ## add the vaccine data for Switzerland made available by Our world in Data 
        vac = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')
        
        # selecting the switzerland data 
        vac = vac.loc[vac.iso_code == 'CHE']
        vac.index = pd.to_datetime(vac.date)
        
        # selecting only the column with vaccinations per hundred 
        vac = vac[['total_vaccinations_per_hundred']]
        
        vac = vac.fillna(0)
        
        if vac.total_vaccinations_per_hundred[-1] == 0:
            vac.total_vaccinations_per_hundred[-1] = vac.total_vaccinations_per_hundred[-2]
    
        
        df['vac_all'] = vac.total_vaccinations_per_hundred
        
    
    # filling the NaN values by zero 
    df = df.fillna(0)
    
    if smooth == True:
        df = df.rolling(window = 7).mean()
        
        df = df.dropna()

        
    return df 


#@st.cache 
def get_canton_data(curve, canton, ini_date = None):
    '''
    This function provide a dataframe for the curve selected in the param curve and
    the canton selected in the param canton
    
    param curve: strin. One of the following options are accepted: ['cases', 'casesVaccPersons', 'covidCertificates', 'death',
                                                             'deathVaccPersons', 'hosp', 'hospCapacity', 'hospVaccPersons',
                                                             'intCases', 're', 'test', 'testPcrAntigen', 'virusVariantsWgs']
    param canton: array with all the cantons of interest.
    return dataframe
    '''
    
    # dictionary with the columns that will be used for each curve. 
    dict_cols = {'cases':['geoRegion','datum','entries'], 'test': ['geoRegion','datum','entries', 'entries_pos'],
                'hosp':['geoRegion','datum','entries', 'sumTotal'], 'hospcapacity':['geoRegion','date','ICU_Covid19Patients', 'Total_Covid19Patients'], 're': ['geoRegion', 'date','median_R_mean']
                }
    
    # getting the data from the databank
    df = pd.read_sql_table(f'foph_{curve}', engine_public, schema = 'switzerland', columns = dict_cols[curve])
    
    # filtering by cantons
    df = df.loc[ (df.geoRegion.isin(canton))]
    
    if (curve == 're') | (curve =='hospcapacity'):
        df.index = pd.to_datetime(df.date)
        
    else: 
        df.index = pd.to_datetime(df.datum)
        df = df.sort_index() 
        
    if ini_date != None: 
        df = df[ini_date:]
    
    return df

def rolling_predictions_ngboost(target_name, data,hosp_as_predict, ini_date = '2020-03-01',split = 0.25, horizon_forecast = 14, maxlag=14):
    '''Train multiple models. One for each prediction delta
    :param df_train: training dataframe 
    :param horizon_forecast: Horizon (maximum forecast range)
    :param maxlag: number of past days to consider
    :param neighbours: knn parameter
    :return: Prediction'''


    target = data[target_name]

    df_lag = build_lagged_features(copy.deepcopy(data), maxlag=maxlag )

    #print(type(df_lag.index[0]))
    #print(target.index[0])

    ini_date = max(df_lag.index[0],target.index[0], datetime.strptime(ini_date, "%Y-%m-%d"))

    df_lag = df_lag[ini_date:]
    target = target[ini_date:]
    df_lag = df_lag.dropna()



    # remove the target column and columns related with the day that we want to predict 
    df_lag = df_lag.drop(data.columns, axis = 1)
    
    for i in df_lag.columns:
        if (i.startswith('hosp') == True) and (i.endswith('lag1') == True or i.endswith('lag2') == True or i.endswith('lag3') == True ):
            #print(i)
            df_lag = df_lag.drop(i, axis = 1)
            
    df_lag = df_lag.dropna()


    # targets 
    targets = {}

    for T in np.arange(1,horizon_forecast+1,1):
        if T == 1:
            targets[T] = target.shift(-(T - 1))
        else:
            targets[T] = target.shift(-(T - 1))[:-(T - 1)]
#         print(T, len(df_lag), len(fit_target))
#         print(df_lag.index,fit_target.index)



    X_train, X_test, y_train, y_test = train_test_split(df_lag, target, train_size=split, test_size=1 - split, shuffle=False)

    idx = pd.period_range(start=df_lag.index[0], end=df_lag.index[-1], freq='14D')
    
    idx = idx.to_timestamp()

    # predictions 
    preds5 = np.empty((len(idx), horizon_forecast))
    preds50 = np.empty((len(idx), horizon_forecast))
    preds95 = np.empty((len(idx), horizon_forecast))

    
    for T in range(1, horizon_forecast + 1):

        tgt = targets[T][:len(X_train)]
        
        X_train = X_train.dropna()
        tgt = tgt.dropna()
        
        i = 0
        
        while i < len(tgt):
            if tgt[i] <= 0:
                
                tgt[i] = 0.1
            i = i+1
        
        model = NGBRegressor(Base=default_tree_learner, Dist =LogNormal,Score = LogScore,
                             natural_gradient=True,verbose=False, col_sample = 0.9, minibatch_frac = 0.9,
                             n_estimators=260,
        learning_rate=0.1,
        validation_fraction=0.25,
        early_stopping_rounds = 50)
        #tgtt = targets[T][len(X_train)]
        #model5 = lgbm_model(alpha = 0.025)
        #model = lgbm_model(alpha = 0.5)
        #model95 = lgbm_model(alpha = 0.975)

        #model5.fit(X_train, tgt, )
        #model50.fit(X_train, tgt,)
        #model95.fit(X_train, tgt, )
        #print(target_name)
        #print(X_train.columns)
        #model5.fit(X_train.iloc[:int(0.75*len(X_train))], tgt.iloc[:int(0.75*len(X_train))],eval_set = [(X_train.iloc[int(0.75*len(X_train)):], tgt.iloc[int(0.75*len(X_train)):])], eval_metric = ['mse', 'mae'] )
        model.fit(X_train.iloc[:len(X_train)], tgt.iloc[:len(X_train)])
        #model95.fit(X_train.iloc[:int(0.75*len(X_train))], tgt.iloc[:int(0.75*len(X_train))], eval_set = [(X_train.iloc[int(0.75*len(X_train)):], tgt.iloc[int(0.75*len(X_train)):])], eval_metric = ['mse', 'mae'])

        #dump(model, f'saved_models/{estimator}_{target_name}_{T}.joblib')
        
        pred = model.pred_dist(df_lag.loc[idx])
        
        pred50 = pred.median()

        pred5, pred95 = pred.interval(alpha = 0.95)

        preds5[:, (T - 1)] = pred5
        preds50[:, (T - 1)] = pred50
        preds95[:, (T - 1)] = pred95


    # transformando preds em um array
    train_size = len(X_train)
    point = targets[1].index[train_size]

    #pred_window = preds5.shape[1]
    #llist = range(len(targets[1].index) - (preds5.shape[1]))
    
    
    #print(llist)

    y5 = preds5.flatten()
    y50 = preds50.flatten()
    y95 = preds95.flatten()

    x= pd.period_range(start=df_lag.index[1], end=df_lag.index[-1], freq='D').to_timestamp()
    
    x = np.array(x)
    
    y5 = np.array(y5)
    
    y50 = np.array(y50)
    
    y95 = np.array(y95)
    
    target = targets[1]
    
    train_size = len(X_train)
    
    dif = len(x) - len(y5)
        
    if dif <0:
        y5 = y5[:len(y5) + dif]
        y50 = y50[:len(y50) + dif]
        y95 = y95[:len(y95) + dif]
        
    df = pd.DataFrame()
    df['target'] = target[1:]
    df['date'] = x
    df['lower'] = y5
    df['median'] = y50
    df['upper'] = y95
    df['train_size'] = [train_size]*len(df)

    return df

def make_predictions_all_cantons(target_curve_name,  predictors, vaccine = True, smooth= True, hosp_as_predict = False,ini_date = '2020-03-01', title = None):

    '''
    Function to make prediction for all the cantons
    
    Important: 
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients
    
    params target_curve_name: string to indicate the target column of the predictions
    params predictors: variables that  will be used in model 
    params vaccinhttp://localhost:8888/notebooks/Documents/GitHub/swiss_COVID19/ML_models/All_cantons.ipynb#e: It determines if the vaccine data from owid will be used or not 
    params smooth: It determines if data will be smoothed or not 
    params hosp_as_predict: It determines if hosp cruves will be use as predictors or not 
    params ini_date: Determines the beggining of the train dataset
    
    returns: Dataframe with the predictions for all the cantons
    '''
    
    df_all = pd.DataFrame()

    # compute the clusters 
    clusters = [#np.array(['FR', 'GE', 'GR', 'JU', 'NE', 'TI', 'VD', 'VS'])]
np.array(['AG', 'AI', 'AR', 'BE', 'BL', 'BS', 'GL', 'LU', 'NW', 'OW', 'SG',
       'SH', 'SO', 'SZ', 'TG', 'UR', 'ZG', 'ZH'])]

    for cluster in clusters: 
        # getting the data 
        df = get_combined_data(predictors, cluster, vaccine=vaccine, smooth = smooth)
        # filling the nan values with 0
        df = df.fillna(0)
    
        for canton in cluster:
            # apply the model 

            target_name = f'{target_curve_name}_{canton}'


            # get predictions and forecast 
            #date_predsknn, predsknn, targetknn, train_size, date_forecastknn, forecastknn = rolling_predictions(model_knn, 'knn', target_name, df , ini_date = '2021-01-01',split = 0.75,   horizon_forecast = horizon, maxlag=maxlag,)
            # x, y5,y50, y95,  target,train_size, forecast_dates, forecasts5, forecasts50,forecasts95 = rolling_predictions(target_name, df , hosp_as_predict,  ini_date = ini_date,split = 0.75,   horizon_forecast = horizon, maxlag=maxlag)

            df_pred = rolling_predictions_ngboost(target_name, df,hosp_as_predict= True , ini_date = '2020-03-01',split = 0.75, horizon_forecast = 14, maxlag=14)
                
            #fig = plot_predictions(target_curve_name, canton, target, train_size, x, y5,y50, y95, forecast_dates, forecasts5, forecasts50,forecasts95, title, path)

            
            df_all = pd.concat([df_all,df_pred])
            
    return df_all


startTime = time.time()

target_curve_name = 'hosp'
predictors = ['cases', 'hosp', 'test', 'hospcapacity']

df_prev = make_predictions_all_cantons(target_curve_name,  predictors, vaccine = True, smooth= True, hosp_as_predict = True,ini_date = '2020-03-01', title = None)

df_prev.to_csv('ngboost_preds_cantons_2.csv')
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
