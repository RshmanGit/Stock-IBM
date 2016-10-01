from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt,mpld3
from matplotlib import style
import datetime
import time

# Create your views here.
def index(request):
	html = '<h1>Enter the Market code for the company you to predict data of: </h1>\
			<form method="GET" action="http://127.0.0.1:8000/predict/update/">\
				<input type="text" name="cmpnyTag">\
				<input type="submit" name="submit" value="submit">\
			</form>\
			<p>example: ED, SYK, DFS, GERN, GOOGL</p>\
			<p>For more reference: </p><a href="https://www.quandl.com/data/WIKI">Click HERE!!</a>'
	return HttpResponse(html)

def work(request):
	html = '<h1>The Prediction was generated</h1>'
	cmpnyTag = request.GET['cmpnyTag']#styling
	style.use('ggplot')

	df = quandl.get("WIKI/"+cmpnyTag)
	df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
	df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
	df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

	df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
	forecast_col = 'Adj. Close'
	df.fillna(value=-99999, inplace=True)
	forecast_out = int(math.ceil(0.01 * len(df)))
	df['label'] = df[forecast_col].shift(-forecast_out)

	X = np.array(df.drop(['label'], 1))
	X = preprocessing.scale(X)
	X_lately = X[-forecast_out:]
	X = X[:-forecast_out]

	df.dropna(inplace=True)

	y = np.array(df['label'])

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
	clf = LinearRegression(n_jobs=-1)
	clf.fit(X_train, y_train)
	confidence = clf.score(X_test, y_test)

	forecast_set = clf.predict(X_lately)
	df['Forecast'] = np.nan

	last_date = df.iloc[-1].name
	last_unix = last_date.strftime('%s')
	one_day = 86400
	next_unix = int(last_unix) + one_day

	for i in forecast_set:
	    next_date = datetime.datetime.fromtimestamp(next_unix).strftime('%Y-%m-%d')
	    next_unix += 86400
	    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

	fig = plt.figure()
	df['Adj. Close'].plot()
	df['Forecast'].plot()
	plt.legend(loc=4)
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title("Predicted Chart with tested accuracy of: "+str(confidence*100)+" :Using Linear regression")
	g = mpld3.fig_to_html(fig)
	html += g
	html2 = '<p>The zoom in, crawl and home tools will appear once you hover over the chart</p></br>\
			 <a href="http://127.0.0.1:8000/predict/">Go Back</a>'
	html += html2

	return HttpResponse(html)