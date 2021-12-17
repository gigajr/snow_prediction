#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ======================================================================================================================
# Temperateure prediction in a city using Prophet framework
# ======================================================================================================================
# Author:     Julien Ros
# Company:    Qapa
# Date:       9 October 2018
# ======================================================================================================================
import pandas as pd
import numpy as np
import logging
import argparse
import pymysql
import sys
from fbprophet import Prophet
import matplotlib.pylab as plt
from plotly.offline import plot
from plotly import graph_objs as go

logging.basicConfig(level=logging.INFO)


# ======================================================================================================================
def plotly_df(df, title='', filename='sampleplot.html'):
    """Visualize all the dataframe columns as line plots."""
    common_kw = dict(x=df.index, mode='lines')
    data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]
    layout = dict(title=title)
    fig = dict(data=data, layout=layout)
    plot(fig, show_link=False, filename=filename)


# ======================================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Qapa Booking Forecasting')
    parser.add_argument('--mysql_server', help='Mysql server', default='10.5.0.7')
    parser.add_argument('--mysql_db', help='Mysql database', default='qapa_production_fr_fr')
    parser.add_argument('--mysql_user', help='Mysql user', default='qapa-read-only')
    parser.add_argument('--mysql_pass', help='Mysql password', default='aiYau2aejas5laepai4l')


    args = parser.parse_args()
    mysql_server = args.mysql_server
    mysql_db = args.mysql_db
    mysql_user = args.mysql_user
    mysql_pass = args.mysql_pass


    logging.info("Loading from db")
    conn = pymysql.connect(host=mysql_server, user=mysql_user, password=mysql_pass, db=mysql_db)
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute(QUERY)
    train = pd.DataFrame(cursor.fetchall()).astype({
            'date_contract': 'object',
            'total_ht': 'float64',
        })
    cursor.close()
    conn.close()

    train['date_contract'] = pd.to_datetime(train['date_contract'])
    train.loc[(train['total_ht'] == 0), 'total_ht'] = 1
    print(train.head())


    daily_df = train.set_index('date_contract').resample('D').apply(sum)
    weekly_df = daily_df.resample('W').apply(sum)
    monthly_df = daily_df.resample('M').apply(sum)
    print(daily_df.head(n=3))
    plotly_df(daily_df, title='Bookings (daily)', filename='daily.html')
    plotly_df(weekly_df, title='Bookings (weekly)', filename='weekly.html')
    plotly_df(monthly_df, title='Bookings (monthly)', filename='monthly.html')


    fr_holidays = holidays.France(years=[2017,2018,2019,2020,2021,2022,2023])
    fr_holidays_dataframe = pd.DataFrame({
        'holiday': 'fr',
        'ds': pd.to_datetime(list(fr_holidays.keys())),
    })

    m1 = Prophet(holidays=fr_holidays_dataframe)
    df = daily_df.reset_index()
    df.columns = ["ds", "y"]
    m1.fit(df)
    future1 = m1.make_future_dataframe(periods=1095)
    forecast1 = m1.predict(future1)


    daily_forecast_df = forecast1[['ds', 'yhat']].copy()
    print(daily_forecast_df.head())
    daily_forecast_df = daily_forecast_df.set_index('ds')

    print(daily_forecast_df.head())
    forecast1_monthly = daily_forecast_df.resample('M').apply(sum)
    print(forecast1_monthly.head())

    plotly_df(forecast1_monthly, title='Predicted Bookings (monthly)', filename='predicted_monthly.html')



    sys.exit(0)
