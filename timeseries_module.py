#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:14:51 2019

@author: bryanbutler
"""

# file to create all of the time series functions

# import basic libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame


# stats libraries
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tools.eval_measures import mse, rmse
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller



# create augmented Dickey-Fuller test
def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data

    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val

    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")



# create the Durbin-Watson statistic
def dw(data_frame):
    """
    Take in a data frame use OLS to build the residuals

    Returns the Durbin-Watson Statistic, best value = 2.00
    """

    ols_res = OLS(data_frame, np.ones(len(data_frame))).fit()
    return durbin_watson(ols_res.resid)




# apply dw to a model
def get_dw(model):
    """
    Pass in a model, get the D-W statistic on the residuals

    """

    resid = model.resid
    return dw(resid)



# make the model
def build_model(series, p, d, q, S, exog_data, P=None, D=None, Q=None):
    """
    Function to build SARIMAX model

    inputs:

        series = name of the series in the dataframe; should be specified in the following
        df['series_name'], series = 'series_name'

        p,d,q for arima modeling

        S: seasonal lag

        P,D,Q for seasonal modeling

        p,P: autoregressive components

        d,D: differencing components

        q,Q: moving average of error term components

        exog_data = matrix of exogenous variables

    default mode sets seasonal P, D, Q = p,d,Q

    Output;

    SARIMAX model results
    """
    if P is None:
        P = p
    if D is None:
        D = d
    if Q is None:
        Q = q
    model = SARIMAX(series, order=(p,d,q),
                    seasonal_order=(P,D,Q,S),
                    exog=exog_data,
                    enforce_invertibility=True)
    results = model.fit()
    return results



# make the backtest model to compare
def backtest_model(model_results, exog_data, train, end, start=1, name='Backtest Model'):
        """
        Create backtest values for the model to test against historical actual

        Inputs:
            model_results = used for the name of the model from build model

            exog_data: matrix of exogenous variables

            start: starting lag to model against, usually 1, not zero

            train: name of the training set to get the lengths from

            end: default to the length of the training set - 1 (len(train)-1)

            """
        results = model_results.predict(start=1,
                                        end=end,
                                        exog=exog_data,
                                        type='levels').rename(name)
        return results





# make the predictions from start to end
def make_predictions(model_results, model_name, start, end, exog_data):
    """
    Predict model results given a start end end

    Inputs:
        model_result: name of model variable

        model_name: name to be used for model results

        start: where to start sequence at (integer)

        end: where to end predictions (integer)

        exog_data: matrix of exogenous variables
    """
    predictions = model_results.predict(start=start,
                                        end=end,
                                        exog=exog_data).rename(model_name)
    return predictions



# check full fit
# make a plot of model fit
def plot_fit(series, backtest_model, train):
    """
    Make a plot that compares the model to the actual series via backtest

    Inputs:
        series: name of series of actual results

        backtest_model: the name of the results of the backtest model;
        usually backtest or xxx_backtest

        train: name of the training dataframe
    """
    # series is training data
    ax = series.plot(figsize=(16,8), legend=True)
    ax1 = backtest_model.plot(legend=True)
    ax.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    # add the holiday indicators in
    for day in train.query('holiday==1').index:
        # add in a vertical line there
        ax.axvline(x=day, color='red', linestyle='--', alpha=.25);




# test the predictions
# make a plot of model fit

def prediction_plot(series, predictions, holiday_df):
    """
    Plot the predictions vs the actual series

    Inputs:
        series: the name of the actual data series

        predictions: name of predictions data series

    Output:
        returns a plot of both series
    """
    # series is usually the test data
    ax = series.plot(figsize=(16,8), legend=True)
    ax1 = predictions.plot(legend=True)
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    for day in holiday_df.query('holiday==1').index:
        # add in a vertical line there
        ax.axvline(x=day, color='red', linestyle='--', alpha=.25);



# make a dataframe to compare predictions
def compare_results(test_data, predictions):
    """
    Return a new dataframe of test results and predictions

    Inputs:
        test_data: the test data set (series)

        predictions: series of prdictions

    Output:
        dataframe containing the two series
    """
    new_df = pd.concat([test_data, predictions], axis = 1 )
    return new_df



# calculate periodic error
def error_calcs(df, modeled, actual):
    """
    Calculate the error at each prediction point

    Input:
        df: name of the dataframe created from the compare results

    Output:
        Error: column calculating the actual error between the prediction and actual;
        calculation is based on actual - predicted


        Percent: converts the error to a percentage of the actual

    Returns a dataframe styled with clean formatting, and 2 decimal points for percent
    """
    df['Error'] = actual - modeled
    df['Percent'] = df['Error']/actual* 100
    return df.style.format({list_cols[0]: "{:,.0f}".format,
                            list_cols[1]: "{:,.0f}".format,
                            list_cols[2]: "{:,.0f}".format,
                            list_cols[3]: "{:,.2f}".format})




# get a roll up of the total error
def calculate_total_error(actual, predictions):
    """
    Calculate root mean square error (RMSE), mean and error as a percentage of mean

    Inputs:
        actual: values of actual data series

        predictions: values of prediction data series

    Outputs:
        root mean squared error of the two series

        mean of the actual series

        percent: percentage of rmse of the actual mean


    Means and errors are formatted as integers

    Percent is formatted as one decimal point
    """
    error = rmse(actual, predictions)
    print(f'{error:.0f}', 'RMSE')

    CancMean = actual.mean()
    print(f'{CancMean:.0f}', 'Mean of Rides')

    percent = error/CancMean*100
    print(f'{percent:.1f}', '% Error')



# create the confidence intervals around the forecasts
def get_conf_interval(model, actual, steps_ahead, predictions, exog_data, alpha = 0.05):
    """
    Create upper and lower confidence intervals around the predictions

    Inputs:
        model: the model to get the forecasts

        actual: actual data series

        steps_ahead: number of steps ahead that model is forecasting (integer)

        predictions: prediction series

        exog_data: matrix of exogenous data, default is normally test[['holiday']]

        alpha: amount in the tails, at 0.05, 95% confidence intervals

    Output:
        Dataframe with actual, predictions, lower confidence interval of predictions and
        upper confidenc interval of predictions

    """
    predictions_int = model.get_forecast(steps=steps_ahead,exog=exog_data, alpha = alpha)

    conf_df = pd.concat([actual,predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    conf_df = conf_df.rename(columns={0: 'Predictions', 1: 'Lower CI', 2: 'Upper CI'})

    return conf_df.style.format("{:,.0f}")



# resample for monthly
def monthly_rollup(model, actual, steps_ahead, predictions, exog_data):
    """
    Resample weekly forecasts into monthly forecasts - used for in series forecasting against test data

    Inputs:
        model: name of model

        actual: name of the series representing actual data

        steps_ahead: number of steps ahead in forecast horizon

        predictions: name of series representing the predictions

        exog_data: matrix of exogenous variable, usually test[['holiday]]

    Output:
        returns a dataframe of forecast and interval values rolled up to monthly values
        Values are formmatted
    """
    predictions_int = model.get_forecast(steps=steps_ahead,exog=exog_data)
    conf_df = pd.concat([actual,predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    conf_df = conf_df.rename(columns={0: 'Predictions', 1: 'Lower CI', 2: 'Upper CI'})

    # convert it to a dataFrame
    conf_df = pd.DataFrame(conf_df)

    # resample for monthly frequency
    conf_df = conf_df.resample('M').agg({np.sum})

    # return pd.DataFrame(conf_df)
    return conf_df.style.format("{:,.0f}")



#### OUT OF SAMPLE FUNCTIONS ###########


def get_oos_conf_interval(model, steps_ahead, exog_data, alpha=0.05):
    """
    Get confidence intervals for out of sample forecasts

    This differs from the other function in that it

    Inputs:
        model: name of model

        steps_ahead: nupber of steps ahead for forecasting (integer)

        exog_data: matrix of out of series exogenous data for oos;
        generall oos[['holiday']]

        alpha: amount in the tails, at 0.05, 95% confidence intervals


    Outputs:

        Returns a style object with predictions, lower confidence interval, upper confidence interval

    """
    predictions_int = model.get_forecast(steps=steps_ahead,exog=exog_data, alpha = alpha)


    conf_df = pd.concat([predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    conf_df = conf_df.rename(columns={0: 'Predictions', 1: 'Lower CI', 2: 'Upper CI'})

    return conf_df.style.format("{:,.0f}")


# create a dataframe for oos plotting
def make_oos_plot_df(model, steps_ahead, exog_data):
    """
    Create a dataframe of values to be used in plotting the out of sequence values and intervals

    Inputs:
        model: name of model used to develop forecasts

        steps_ahead: number of steps ahead for forecasting

        exog_data: matrix of exogenous variables, usually oos_exog or related matrix

    Output:
        returns a dataframe of predictions, lower confidence interval and upper confidence interval;
        since this is out of series (oos), no actuals are available
    """
    predictions_int = model.get_forecast(steps=steps_ahead,exog=exog_data)
    conf_df = pd.concat([predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    conf_df = conf_df.rename(columns={0: 'Predictions', 1: 'Lower CI', 2: 'Upper CI'})

    # convert it to a dataFrame
    conf_df = pd.DataFrame(conf_df)

    return conf_df




def plot_oos(conf_df, df, series, backtest, fut_exog, start_date='2019-07-01'):
    """
    Make a plot of the series, out of sample forecasts and shaded confidence intervals

    Inputs:
        conf_df: name of the dataframe created using the make_oos_plot_df

        df: name of the dataframe that holds the series of actual values

        series: name of the series of actual values

        backtest: name of the series holding the backtest resultsu

        fut_exog: matrix of future dates and exogenous variables

        start_date: starting date to show the window of actual values; defaults to mid-year to keep
        the plot well-scaled

    Output:
        returns a plot with the actual series, the modeled series, confidence interval and shaded
        region within the conidence intervals

    """
    # make a plot of model fit
    # color = 'skyblue'

    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(111)

    # set the x value to the index
    x = conf_df.index.values

    # get the confidence interval series
    upper = conf_df['upper ' + series]
    lower = conf_df['lower ' + series]

    # add the actual data starting at the start_date
    ax = df[series].loc[start_date:].plot(figsize=(16,8),
                                          legend=True,
                                          label = 'Actual ' + series,
                                          linewidth=2)

    # add in the backtest series
    ax0 = backtest.loc[start_date:].plot(color = 'orange', label = 'Model backtest')

    # plot the predictions
    ax1 = conf_df['Predictions'].plot(color = 'orange',label = 'Predicted ' + series )

    # plot the uper and lower confidence bounds
    ax2 = upper.plot(color = 'grey', label = 'Upper CI')
    ax3 = lower.plot(color = 'grey', label = 'Lower CI')

    # plot the legend for the first plot
    plt.legend(loc = 'lower left', fontsize = 12)


    # fill between the conf intervals
    plt.fill_between(x, lower, upper, color='grey', alpha='0.2')

    # add the holiday indicators in for the actuals
    for day in df.query('holiday==1').index:
        # add in a vertical line there
        ax.axvline(x=day, color='red', alpha=.2)

    # add the holidays in for the ooos
    for day in fut_exog.query('holiday==1').index:
        # add in a vertical line there
        ax.axvline(x=day, color='red', alpha=.2)

    # format the y values with commas
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    plt.show();




def oos_monthly_rollup(model, steps_ahead, exog_data, series, add_date=False):
    """
    Create a dataframe for the out of sequence monthly rollup of forecasts

    Inputs:

        model: name of model for forecasting

        steps_ahead: number of steps ahead for weekly forecast

        exog_data: matrix of time series with exogenous data

        add_date: boolean value to add the current date as a column for date of forecast

    Output:
        returns a formatted data frame of out of series predictions, lower confidence interval,
        upper confidence interval, and date of fcast if add_date set to True
    """
    from datetime import datetime as dt
    predictions_int = model.get_forecast(steps=steps_ahead,exog=exog_data)
    conf_df = pd.concat([predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    conf_df = conf_df.rename(columns={0: 'Predictions', 1: 'Lower CI', 2: 'Upper CI'})

    # convert it to a dataFrame
    conf_df = pd.DataFrame(conf_df)

    # resample for monthly frequency
    conf_df = conf_df.resample('M').agg({np.sum})
    conf_df.columns = [series, 'lower_' + series, 'upper_' + series]
    list_cols = conf_df.columns

    # add date of forecast if add_date = true
    if add_date:
        conf_df['fcast_date'] = dt.now().date()

    # return pd.DataFrame(conf_df)
    # this returns a syler object, need a dataframe
    # return conf_df.style.format({list_cols[0]: "{:,.0f}".format,
    #                             list_cols[1]: "{:,.0f}".format,
    #                             list_cols[2]: "{:,.0f}".format})
    return conf_df





def cpm_monthly_rollup(model, steps_ahead, exog_data, series, add_date=False):
    """
    Create a dataframe for the out of sequence monthly rollup of forecasts

    Inputs:

        model: name of model for forecasting

        steps_ahead: number of steps ahead for weekly forecast

        exog_data: matrix of time series with exogenous data

        add_date: boolean value to add the current date as a column for date of forecast

    Output:
        returns a formatted data frame of out of series predictions, lower confidence interval,
        upper confidence interval, and date of fcast if add_date set to True
    """
    from datetime import datetime as dt
    predictions_int = model.get_forecast(steps=steps_ahead,exog=exog_data)
    conf_df = pd.concat([predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    conf_df = conf_df.rename(columns={0: 'Predictions', 1: 'Lower CI', 2: 'Upper CI'})

    # convert it to a dataFrame
    conf_df = pd.DataFrame(conf_df)

    # resample for monthly frequency
    conf_df = conf_df.resample('M').agg({np.mean})
    conf_df.columns = [series, 'lower_' + series, 'upper_' + series]
    list_cols = conf_df.columns

    # add date of forecast if add_date = true
    if add_date:
        conf_df['fcast_date'] = dt.now().date()

    # return pd.DataFrame(conf_df)
    # this returns a syler object, need a dataframe
    # return conf_df.style.format({list_cols[0]: "{:,.0f}".format,
    #                             list_cols[1]: "{:,.0f}".format,
    #                             list_cols[2]: "{:,.0f}".format})
    return conf_df




def oos_weekly_df(model, steps_ahead, exog_data, series, add_date=False):
    """
    Create a dataframe for the weekly predictions values to be imported and used later

    Inputs:

        model: name of model for forecasting

        steps_ahead: number of steps ahead for weekly forecast

        exog_data: matrix of time series with exogenous data

        add_date: boolean value to add the current date as a column for date of forecast

    Output:
        returns a formatted data frame of out of series predictions, lower confidence interval,
        upper confidence interval, and date of fcast if add_date set to True
    """
    from datetime import datetime as dt
    predictions_int = model.get_forecast(steps=steps_ahead,exog=exog_data)
    conf_df = pd.concat([predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    conf_df = conf_df.rename(columns={0: 'Predictions', 1: 'Lower CI', 2: 'Upper CI'})

    # convert it to a dataFrame
    conf_df = pd.DataFrame(conf_df)

    # rename the columns
    conf_df.columns = [series, 'lower_' + series, 'upper_' + series]

    # add date of forecast if add_date = true
    if add_date:
        conf_df['fcast_date'] = dt.now().date()

    return conf_df











