# Deep Q-Trader

This project implements Q-learning for short-term stock trading, based on the version of [edwardhdlu](https://github.com/edwardhdlu/q-trader/tree/master). The model analyzes n-day windows of closing prices to decide whether the best action at any given moment is to buy, sell, or hold. 

Due to its focus on short-term states, the model isn’t very effective at making long-term trend decisions, but it excels at predicting short-term peaks and troughs

## Results

The model behavior is not excelling for "long time investments", but it is accurate at predicting peaks where to sell. One model is always trained on only one stock. Applying them to different stocks shows generalization capabilities. Nevertheless, not always the profit is high or positive. 

![Model1](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Model_GSPC_reward_target10_on_GSPC_2011.png)

![Model2](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Model_GSPC_reward_target6_on_GSPC_2011.png)

![Generalization](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Model_GSPC_reward_target6_on_SundP_2024.png)

![Model3](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Model_GSPC_reward_target8_on_GSPC_2011.png)
![Generalization_positive](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Model_GSPC_reward_target8_on_SundP_2024.png)


## Getting the data 

For model training, a csv file containing the collumns Date, Open, High, Low, Close of a chosen stock for at least a year. Data can be retrieved from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) and stored in `data/`.

## Starting 
```
mkdir model
python train ^GSPC 10 1000
```