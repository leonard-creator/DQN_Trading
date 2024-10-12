# Deep Q-Trader

This project implements Q-learning for short-term stock trading, based on the version of [edwardhdlu](https://github.com/edwardhdlu/q-trader/tree/master). The model analyzes n-day windows of closing prices to decide whether the best action at any given moment is to buy, sell, or hold. 

Due to its focus on short-term states, the model isn’t very effective at making long-term trend decisions, but it excels at predicting short-term peaks and troughs

## Results

The model does achieve notable profits, but is not consistent and can not generalize well accross different datasets yet.

Graphs of the results will be shown soon... 

## Getting the data 

For model training, a csv file containing the collumns Date, Open, High, Low, Close of a chosen stock for at least a year. Data can be retrieved from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) and stored in `data/`.

## Starting 
```
mkdir model
python train ^GSPC 10 1000
```