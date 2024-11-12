# Deep Q-Trader

This project implements Q-learning for short-term stock trading, based on the version of [edwardhdlu](https://github.com/edwardhdlu/q-trader/tree/master). The Model analyzes n-day windows of closing prices to decide whether the best action at any given moment is to buy, sell, or hold. 

Due to its focus on short-term states, the Agent isn’t very effective at making long-term trend decisions, but it excels at predicting short-term peaks and troughs

## Results

The Agent behavior is not ideal for "long time investments", but it is accurate at predicting peaks where to sell. One Agent is always trained on only one stock. Applying them to different stocks shows generalization capabilities. Nevertheless, not always the profit is high or positive. 

### Google
Agent trained on 5 years Google stock one complete episode

![Agent1](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Google0_googletest.png)

### Google
Agent trained on 5 years Google stock two complete episode
![Agent1.2](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Google1_googletest.png)

### Google 
Agent trained on 5 years Google stock one complete episode, tested on 2024 S&P data
![Agent1_generailzation](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Google0_SundP_test.png)

### Google
Agent trained on 5 years Google stock one complete episode, tested on 2024 MSCI-World test data
![Agent1_generailzation](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Google0_iSharesMSCIWorld_test.png)


### S&P small
Agent trained on 2019-2022 S&P index, tested on 2024 S&P data 
![Agent3](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/smallSP2_SundPtest.png)

### S&P small
Agent trained on 2019-2022 S&P index, tested on 2024 Google-test data
![Agent3](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/smallSP2_Google_test.png)

### S&P small
Agent trained on 2019-2022 S&P index, tested on 2024 MSCI-World test data
![Agent3](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/smallSP2_iSharesMSCIWorld-test.png)



## Sourcing the data 

For model training, a csv file containing the collumns Date, Open, High, Low, Close of a chosen stock for at least a year. Data can be retrieved from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) and stored in `data/`.

## Training 
```
mkdir model
python train ^GSPC 10 1000
```
## Evaluating 

```
python evaluate.py [stock dataset name] [Agent name] [verbose(yes/no)]
```

The Agent is applied on the dataset, presenting the final profit. A graph will be created that indicates the stock price on the y-axis and the trading days on the x-axis. Green and red rectangular are placed to indicate buy and cell operations taken by the Agent. The graph can be viewed interactively (zoom in or moved) and saved. 