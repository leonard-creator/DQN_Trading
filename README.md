# Deep Q-Trader

This project implements Q-learning for short-term stock trading, based on the version of [edwardhdlu](https://github.com/edwardhdlu/q-trader/tree/master). The model analyzes n-day windows of closing prices to decide whether the best action at any given moment is to buy, sell, or hold. 

Due to its focus on short-term states, the model isn’t very effective at making long-term trend decisions, but it excels at predicting short-term peaks and troughs

## Results

The model behavior is not ideal for "long time investments", but it is accurate at predicting peaks where to sell. One model is always trained on only one stock. Applying them to different stocks shows generalization capabilities. Nevertheless, not always the profit is high or positive. 

### Model 1 
![Model1](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Model_GSPC_reward_target10_on_GSPC_2011.png)

### Model 1 version2
![Model2](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Model_GSPC_reward_target6_on_GSPC_2011.png)

### Model 1 version2 on different stock 
![Generalization](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Model_GSPC_reward_target6_on_SundP_2024.png)

### Model 1 version3 
![Model3](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Model_GSPC_reward_target8_on_GSPC_2011.png)

### Model 1 version3 on different stock
![Generalization_positive](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Model_GSPC_reward_target8_on_SundP_2024.png)


## Sourcing the data 

For model training, a csv file containing the collumns Date, Open, High, Low, Close of a chosen stock for at least a year. Data can be retrieved from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) and stored in `data/`.

## Training 
```
mkdir model
python train ^GSPC 10 1000
```
## Evaluating 

```
python evaluate.py [stock dataset name] [model name] [verbose(yes/no)]
```

The model is applied on the dataset, presenting the final profit. A graph will be created that indicates the stock price on the y-axis and the trading days on the x-axis. Green and red rectangular are placed to indicate buy and cell operations taken by the model. The graph can be viewed interactively (zoom in or moved) and saved. 