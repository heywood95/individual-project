#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from env import host, username, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from scipy import stats
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

def prep_world_govt(df):
    '''Prepares acquired world government data for exploration'''
    
    # drop the null values
    df = df.dropna()
    
    # replace the '..' placeholders with a zero, the .. is a placeholder for a zero value
    df = df.replace('..', 0)
    
    # turn the numerical columns that are objects into floats
    df['Control of Corruption: Percentile Rank [CC.PER.RNK]'] = df['Control of Corruption: Percentile Rank [CC.PER.RNK]'].astype('float')
    df['Government Effectiveness: Percentile Rank [GE.PER.RNK]'] = df['Government Effectiveness: Percentile Rank [GE.PER.RNK]'].astype('float')
    df['Political Stability and Absence of Violence/Terrorism: Percentile Rank [PV.PER.RNK]'] = df['Political Stability and Absence of Violence/Terrorism: Percentile Rank [PV.PER.RNK]'].astype('float')
    df['Regulatory Quality: Percentile Rank [RQ.PER.RNK]'] = df['Regulatory Quality: Percentile Rank [RQ.PER.RNK]'].astype('float')
    df['Rule of Law: Percentile Rank [RL.PER.RNK]'] = df['Rule of Law: Percentile Rank [RL.PER.RNK]'].astype('float')
    df['Voice and Accountability: Percentile Rank [VA.PER.RNK]'] = df['Voice and Accountability: Percentile Rank [VA.PER.RNK]'].astype('float')

    # drop unnecessary and redundant columns
    df = df.drop(columns=['Country Code', 'Time Code', 'Country Name'])
    
    # rename confusing and exceedingly long column names
    df = df.rename(columns={'Time': 'date', 
                            'Control of Corruption: Percentile Rank [CC.PER.RNK]': 'control_corruption',
                            'Government Effectiveness: Percentile Rank [GE.PER.RNK]': 'govt_effective',
                             'Political Stability and Absence of Violence/Terrorism: Percentile Rank [PV.PER.RNK]': 'political_stability',
                            'Regulatory Quality: Percentile Rank [RQ.PER.RNK]': 'regulatory_quality',
                            'Rule of Law: Percentile Rank [RL.PER.RNK]': 'rule_of_law',
                            'Voice and Accountability: Percentile Rank [VA.PER.RNK]': 'voice_accountability'})
    
    # change the date column from float to an integer
    df['date'] = df['date'].astype('int')
    
    # create a new column 'overall_govt' that includes the six features and the mean
    # of the sum of those features
    df['overall_govt'] = df['control_corruption'] + df['govt_effective'] + df['political_stability'] + df['regulatory_quality'] + df['rule_of_law'] + df['voice_accountability']
    df['overall_govt'] = df['overall_govt'] / 6
    
    # split the data
    train, test = train_test_split(df, train_size = 0.8, random_state = 123)
    train, validate = train_test_split(train, train_size = 0.7, random_state = 123)
    
    return train, validate, test

def split_data(df):
    '''
    This function takes in a dataframe and splits the data,
    returning three pandas dataframes, train, test, and validate
    '''
    #create train_validate and test datasets
    train, test = train_test_split(df, train_size = 0.8, random_state = 123)
    #create train and validate datasets
    train, validate = train_test_split(train, train_size = 0.7, random_state = 123)
  
    return train, validate, test 


#create a function to isolate the target variable
def X_y_split(df, target):
    '''
    This function takes in a dataframe and a target variable
    Then it returns the X_train, y_train, X_validate, y_validate, X_test, y_test
    and a print statement with the shape of the new dataframes
    '''  
    train, validate, test = split_data(df)

    X_train = train.drop(columns= [target])
    y_train = train[target]

    X_validate = validate.drop(columns= [target])
    y_validate = validate[target]

    X_test = test.drop(columns= [target])
    y_test = test[target]
        
    # Have function print datasets shape
    print(f'X_train -> {X_train.shape}')
    print(f'X_validate -> {X_validate.shape}')
    print(f'X_test -> {X_test.shape}')  
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test



def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['overall_govt', 'control_corruption', 'govt_effective', 'political_stability', 'regulatory_quality', 'rule_of_law', 'voice_accountability'], return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    from sklearn.preprocessing import MinMaxScaler
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),                              columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),                        columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),                                columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled


def rfe(X, y, n):
    '''
    This function takes in the features, target variable 
    and number of top features desired and returns a dataframe with
    the features ranked
    '''
    
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select=n)
    rfe.fit(X, y)
    ranks = rfe.ranking_
    columns = X.columns.tolist()
    feature_ranks = pd.DataFrame({'ranking': ranks, 'feature': columns})
    return feature_ranks.sort_values('ranking')

def visual_correlations(df):
    '''
    This function creates a heatmap of the features
    '''
    
    count_var = ['overall_govt', 'control_corruption', 'govt_effective', 'political_stability', 'regulatory_quality', 'rule_of_law', 'voice_accountability']

    train_corr = df[count_var].corr()
    
    plt.title('Strength of Correlation with overall_govt')
    
    sns.heatmap(train_corr, annot=False, annot_kws={"size": 1}, linewidths=2, linecolor='yellow', yticklabels=14)
    
    return plt.show()



def eval_result(df):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    
    alpha = 0.05
    r, p = stats.spearmanr(df.rule_of_law, df.overall_govt)
    r = round(r, 2)
    
    if p < alpha:
        print("Reject the null hypothesis.")
        print("p-value=", p, "r=", r)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    
def eval_result2(df):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    
    alpha = 0.05
    r, p = stats.spearmanr(df.control_corruption, df.overall_govt)
    r = round(r, 2)
    
    if p < alpha:
        print("Reject the null hypothesis.")
        print("p-value =", p, "r =", r)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")

        
def eval_result3(df):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    
    alpha = 0.05
    r, p = stats.spearmanr(df.regulatory_quality, df.overall_govt)
    r = round(r, 2)
    
    if p < alpha:
        print("Reject the null hypothesis.")
        print("p-value =", p, "r =", r)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null") 

        
        
def eval_result4(df):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    
    alpha = 0.05
    r, p = stats.spearmanr(df.political_stability, df.overall_govt)
    r = round(r, 2)
    
    if p < alpha:
        print("Reject the null hypothesis.")
        print("p-value =", p, "r =", r)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
        
        
        
def eval_result5(df):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    
    alpha = 0.05
    r, p = stats.spearmanr(df.voice_accountability, df.overall_govt)
    r = round(r, 2)
    
    if p < alpha:
        print("Reject the null hypothesis.")
        print("p-value =", p, "r =", r)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
        
        
def eval_result6(df):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    
    alpha = 0.05
    r, p = stats.spearmanr(df.govt_effective, df.overall_govt)
    r = round(r, 2)
    
    if p < alpha:
        print("Reject the null hypothesis.")
        print("p-value =", p, "r =", r)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
        
        

def baseline(yt, yv):
    '''
    This function takes in y_train and y_validate and prints the mean and meadian 
    baselines.
    '''
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    yt = pd.DataFrame(yt)
    yv = pd.DataFrame(yv)

    # 1. Predict G3_pred_mean
    overall_govt_pred_mean = yt['overall_govt'].mean()
    yt['overall_govt_pred_mean'] = overall_govt_pred_mean
    yv['overall_govt_pred_mean'] = overall_govt_pred_mean

    # 2. compute G3_pred_median
    overall_govt_pred_median = yt['overall_govt'].median()
    yt['overall_govt_pred_median'] = overall_govt_pred_median
    yv['overall_govt_pred_median'] = overall_govt_pred_median

    # 3. RMSE of G3_pred_mean
    rmse_train = mean_squared_error(yt.overall_govt, yt.overall_govt_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(yv.overall_govt, yv.overall_govt_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    # 4. RMSE of G3_pred_median
    rmse_train = mean_squared_error(yt.overall_govt, yt.overall_govt_pred_median)**(1/2)
    rmse_validate = mean_squared_error(yv.overall_govt, yv.overall_govt_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    
    
def linear_reg_model(Xt, yt, yv, Xv):
    '''
    This function creates, fits, and predicts the RMSE for a
    LinearRegression model and outputs the results.
    '''
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    yt = pd.DataFrame(yt)
    yv = pd.DataFrame(yv)
    
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(Xt, yt.overall_govt)

    # predict train
    yt['overall_govt_pred_lm'] = lm.predict(Xt)

    # evaluate: rmse
    rmse_train = mean_squared_error(yt.overall_govt, yt.overall_govt_pred_lm)**(1/2)

    # predict validate
    yv['overall_govt_pred_lm'] = lm.predict(Xv)

    # evaluate: rmse
    rmse_validate = mean_squared_error(yv.overall_govt, yv.overall_govt_pred_lm)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)

    
    
    
def tweedie_model(Xt, yt, yv, Xv):
    '''
    This function creates, fits, and predicts the RMSE for a
    Lasso-Lars model and outputs the results.
    '''
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    yt = pd.DataFrame(yt)
    yv = pd.DataFrame(yv)
    
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(Xt, yt.overall_govt)

    # predict train
    yt['overall_govt_pred_glm'] = glm.predict(Xt)

    # evaluate: rmse
    rmse_train = mean_squared_error(yt.overall_govt, yt.overall_govt_pred_glm)**(1/2)

    # predict validate
    yv['overall_govt_pred_glm'] = glm.predict(Xv)

    # evaluate: rmse
    rmse_validate = mean_squared_error(yv.overall_govt, yv.overall_govt_pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    
    
    
def poly_model(Xt, yt, yv, Xv):
    '''
    This function creates, fits, and predicts the RMSE for a
    polynomial model and outputs the results.
    '''
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    yt = pd.DataFrame(yt)
    yv = pd.DataFrame(yv)
    
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(Xt)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(Xv)
    X_test_degree2 = pf.transform(Xt)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, yt.overall_govt)

    # predict train
    yt['overall_govt_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(yt.overall_govt, yt.overall_govt_pred_lm2)**(1/2)

    # predict validate
    yv['overall_govt_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(yv.overall_govt, yv.overall_govt_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)



    
    
    
def lr_test_model(Xtest, ytest, Xt, yt):
    '''
    This function creates, fits, and predicts the RMSE for a
    polynomial test model and outputs the results.
    '''
    
    # We need y_test to be dataframes to append the new columns with predicted values. 
    ytest = pd.DataFrame(ytest)
    yt = pd.DataFrame(yt)

    # predict on test
    lm = LinearRegression(normalize=True)
    lm.fit(Xt, yt.overall_govt)
    ytest['overall_govt_pred_lm'] = lm.predict(Xtest)

    # evaluate: rmse
    rmse_test = mean_squared_error(ytest.overall_govt, ytest.overall_govt_pred_lm)**(1/2)

    print("RMSE for OLS Model using LinearRegression\nOut-of-Sample Performance: ", rmse_test)

    
    
def rule_of_law_visual(df):
    '''
    This function creates a swarmplot for rule of law
    and overall governance.
    '''

    binned = pd.cut(x = df.rule_of_law, bins = 4)
    binned2 = pd.cut(x = df.overall_govt, bins = 4)
    
    sns.swarmplot(x=binned, y='overall_govt', data=df, palette=['red', 'orange', 'blue', 'green'])
    plt.title('Rule of Law has the Strongest Correlation')
    plt.xlabel('Rule of Law')
    plt.ylabel('Overall Governance')
    plt.xticks([])
    return plt.show()



def control_corruption_visual(df):
    '''
    This function creates a swarmplot of controlling corruption
    and overall governance.
    '''
    
    binned3 = pd.cut(x = df.control_corruption, bins = 4)
    binned4 = pd.cut(x = df.overall_govt, bins = 4)
    
    sns.swarmplot(x=binned3, y='overall_govt', data=df, palette=['red', 'orange', 'blue', 'green'])
    plt.title('Controlling Corruption is a Part of Overall Governance')
    plt.xlabel('Controlling Corruption')
    plt.ylabel('Overall Governance')
    plt.xticks([])
    return plt.show()



def stability_quality_overall_visual(df):
    '''
    This function creates a relplot and countplot of political stability,
    regulatory quality, and overall governance.
    '''
    
    binned5 = pd.cut(x = df.regulatory_quality, bins = 4)
    binned6 = pd.cut(x = df.political_stability, bins = 4)
    binned7 = pd.cut(x = df.overall_govt, bins = 4, labels=['Poor', 'Below Avgerage', 'Avgerage', 'Above Average'])
    
    sns.relplot(x='regulatory_quality', y='political_stability', legend='auto', data=df, hue=binned7, palette=['red', 'orange', 'blue', 'green'])
    plt.xlabel('Regulatory Quality')
    plt.ylabel('Political Stability')
    plt.title('Political Stability May Not be Required')
    plt.show()

    ax2 = sns.countplot(x=binned5, hue=binned6, data=df, palette=['red', 'orange', 'blue', 'green'])
    plt.xlabel('Regulatory Quality')
    plt.ylabel('Number of Countries')
    plt.xticks([])
    plt.title('Is Political Stability Neccessary for Regulatory Quality?')
    legend_handles, _= ax2.get_legend_handles_labels()
    ax2.legend(legend_handles, ['Poor','Below Average','Avgerage', 'Above Avgerage'], 
          bbox_to_anchor=(1,1), 
          title='Political Stability')

    return plt.show()



def voice_effective_overall_visual(df):
    '''
    This function creates a relplot and countplot of voice and accountability,
    government effectiveness, and overall governance.
    '''
    
    binned8 = pd.cut(x = df.voice_accountability, bins = 4)
    binned9 = pd.cut(x = df.govt_effective, bins = 4)
    binned10 = pd.cut(x = df.overall_govt, bins = 4, labels=['Poor', 'Below Avgerage', 'Avgerage', 'Above Average'])
    
    sns.relplot(x='voice_accountability', y='govt_effective', data=df, legend='auto', hue=binned10, palette=['red', 'orange', 'blue', 'green'])
    plt.xlabel('Voice and Accountability')
    plt.ylabel('Government Effectiveness')
    plt.title('Accountability Need not Apply')
    plt.show()

    ax4 = sns.countplot(x=binned8, hue=binned9, data=df, palette=['red', 'orange', 'blue', 'green'])
    plt.xlabel('Voice and Accountability')
    plt.ylabel('Number of Countries')
    plt.xticks([])
    plt.title('High Levels of Government Effectiveness can Still Have Low Accountability')
    legend_handles, _= ax4.get_legend_handles_labels()
    ax4.legend(legend_handles, ['Poor','Below Average','Avgerage', 'Above Average'], 
          bbox_to_anchor=(1,1), 
          title='Government Effectiveness')

    return plt.show()


def model_compare_visual(yt, yv, Xt, Xv):
    '''
    This function creates a histplot of the three top models and compares those models to
    the actual values.
    '''
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    yt = pd.DataFrame(yt)
    yv = pd.DataFrame(yv)
    
    # create the model object
    lm = LinearRegression(normalize=True)
    glm = TweedieRegressor(power=1, alpha=0)
    pf = PolynomialFeatures(degree=2)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(Xt, yt.overall_govt)
    glm.fit(Xt, yt.overall_govt)
    
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(Xt)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(Xv)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, yt.overall_govt)

    
    # predict validate
    yv['overall_govt_pred_lm'] = lm.predict(Xv)
    yv['overall_govt_pred_glm'] = glm.predict(Xv)
    yv['overall_govt_pred_lm2'] = lm2.predict(X_validate_degree2)
    
    # plot to visualize actual vs predicted. 
    plt.figure(figsize=(16,8))
    plt.hist(yv.overall_govt, color='blue', alpha=.5, label="Actual Govt. Percentage")
    plt.hist(yv.overall_govt_pred_lm, color='red', alpha=.5, label="Model: LinearRegression")
    plt.hist(yv.overall_govt_pred_glm, color='yellow', alpha=.5, label="Model: TweedieRegressor")
    plt.hist(yv.overall_govt_pred_lm2, color='green', alpha=.5, label="Model 2nd degree Polynomial")
    plt.xlabel("Govt. Percentage")
    plt.ylabel("Count")
    plt.title("Comparing the Distribution of Actual Govt. Percentages to Distributions of Predicted Govt. Percentages for the Top Models")
    plt.legend()
    
    return plt.show()



# In[ ]:




