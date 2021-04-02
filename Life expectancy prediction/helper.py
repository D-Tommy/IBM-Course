# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:35:40 2021

@author: Tommy
"""
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def columns_name_correcter(columns):
    cols = columns.values.copy()
    for index, col_name in enumerate(cols):
        cols[index] = col_name.lower().replace(' ', '_').replace('-','_') #removing any upper case
    for index, col_name in enumerate(cols):
        if col_name[0] == '_':
            cols[index] = col_name[1:]
        if col_name[-1] == '_':
            cols[index] = col_name[:-1]
        if col_name[0] == col_name[-1] == '_':
            cols[index] = col_name[1:-1]
    return cols


def outliers_by_country(df, treshold = 1.5):
    """Function that will return a dataframe containing only outliers and the new value they should take
    i.e : the mean of this feature value, for the target country, if this value wasn't accounted"""
    data = df.copy() # copy of our df
    countries = data['country'].unique()
    outliers = pd.DataFrame(columns = data.columns, index = df.index) # preparing the df thta will contain outliers
    outliers_solved = pd.DataFrame(columns = data.columns, index = df.index)

    for country in countries :
        current = data[data['country']==country]

        out_up = pd.DataFrame(current.describe().loc['75%']*treshold).reset_index().rename({'index':'features'}, axis = 1)
        outlier_lim_up = (out_up.pivot(columns = 'features', values = '75%').sum())

        out_low = pd.DataFrame(current.describe().loc['25%']*(1/treshold)).reset_index().rename({'index':'features'}, axis = 1)
        outlier_lim_low = (out_low.pivot(columns = 'features', values = '25%').sum())

        for col in current.columns.drop(['country', 'status']):
            
            column = current[col]
            
            mask_up = [column > outlier_lim_up[col]]
            mask_low = [column < outlier_lim_low[col]]
            filter_tot = [a | b for (a, b) in zip(mask_up, mask_low)]
            neg_filter = filter_tot[0].map({False:True, True:False})
            
            outliers_for_col = column[filter_tot[0]]
            not_outliers_for_col = column[neg_filter]
            
            outliers[col].iloc[outliers_for_col.index] = outliers_for_col.values
            mean_without_outliers = not_outliers_for_col.mean()
            outliers_solved[col].iloc[outliers_for_col.index] = mean_without_outliers 

    return outliers, outliers_solved
    

def remove_NaN_values(df):
    
    data = df.copy()
    #working by country one more time:
    countries = data['country'].unique()
    
    df_NaN_removed = pd.DataFrame(columns = data.columns, index = data.index) # df that will be used to update the main one.
    
    for country in countries :
        current = data[data['country'] == country]
        
        for col in current.columns.drop(['country', 'status']):
            column = current[col]
            nan_values = column.isna().map({False:0, True:1})
                                           
            if (sum(nan_values) == len(column)) :# i.e if eveything is NaN
                df_NaN_removed[col].iloc[column.index] = 0
                    
            elif (sum(nan_values) >= 1) : # i.e  if there's at least one NaN value:
                inverse_mask = nan_values.map({0:True, 1:False})# get all non-NaN index
                mean_non_nan = column[inverse_mask].mean()
                index_nan = nan_values[nan_values == 1].index
                df_NaN_removed[col].iloc[index_nan] = mean_non_nan
                    
            else :
                continue 
    return df_NaN_removed 


def plotRegression(y_pred, y_test):
    sns.scatterplot(y_pred, y_test)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'R2 score = {r2_score(y_pred, y_test)}')
    
    
def plot_highest_coefs(df_coef, n_coefs):        
    data = df_coef.copy() 
    feature_names = data['feature']
    coefs = data['coef']
    
    if n_coefs > 0: # plotting features with a negative impact 
        labels = feature_names[:n_coefs]
        n_firsts = coefs[:n_coefs]
        ax = sns.barplot(x = n_firsts, y = labels)
        ax.set_title(f'{n_coefs} factors reducing the most the life expectancy')
    if n_coefs < 0: # plotting features with a positive impact 
        labels = feature_names[n_coefs:]
        n_lasts = coefs[n_coefs:]
        ax = sns.barplot(x = n_lasts, y = labels)
        ax.set_title(f'{-n_coefs} factors increasing the most the life expectancy')



#%% 
df = pd.read_csv('C:/Users/Tommy/00IBM Course/Pair-reviewed projects/2_Regression/data/Life_Expectancy_Data.csv')



                
                
                
                
