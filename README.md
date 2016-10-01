# Stock-IBM
My project submission for the hackathon conducted by IBM on 1st october 2016

This project is intended to make predictions of stock market prices of a specific company. It uses Linear regression to predict the price based on the data recieved from quandl API. The recieved data stretches from 2004 to the day previous to the one on which the program is executed.

It is a web-application that uses django-python framework as the middleware. The prediction is made using the python library scikit-learn. The quandl API has thousands of different data bases but the one being used here is the Wiki EOD Stock Prices database.

To plot the graph, the python library call matplotlib is used.
