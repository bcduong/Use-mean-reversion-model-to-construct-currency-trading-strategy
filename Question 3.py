import pandas as pd
import numpy as np
import arch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hurst import compute_Hc

# Load data from Excel file
file_path = 'D:/NEOMA/Python/EURUSD.xlsx'
df = pd.read_excel(file_path, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Drop rows with NaN values
df = df.dropna()

# Calculate log returns
df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

# Rescale log returns
rescale_factor = 100
df['Rescaled_Log_Returns'] = rescale_factor * df['Log_Returns']

# Split the data into training and testing sets using train_test_split
train_size = 0.8
train, test = train_test_split(df, test_size=1 - train_size, shuffle=False)

# Calculate the Hurst exponent
hurst_exponent, _, _ = compute_Hc(train['Close'], kind='price', simplified=True)

# Specify GARCH(1,1) model
model = arch.arch_model(train['Rescaled_Log_Returns'].dropna(), vol='Garch', p=1, q=1)

# Fit the GARCH model on the training set
results = model.fit(options={'maxiter': 1000, 'disp': 'on'})

# Get GARCH volatility on the training set
train['GARCH_Volatility'] = results.conditional_volatility

# Introduce volatility adjustments (you can customize the adjustment logic)
volatility_multiplier = 1.5
train['Adjusted_Volatility'] = volatility_multiplier * train['GARCH_Volatility']

# Dynamically adjust threshold based on recent market conditions
threshold_multiplier = 1.0  # Adjust this multiplier based on your strategy and recent conditions
train['Dynamic_Threshold'] = threshold_multiplier * train['Adjusted_Volatility'].rolling(window=20).mean()

# Implement mean reversion trading strategy with dynamic threshold on training set
train['Signal'] = 0
train.loc[(train['Log_Returns'] < 0) & (train['Adjusted_Volatility'] < train['Dynamic_Threshold']), 'Signal'] = 1
train.loc[(train['Log_Returns'] > 0) & (train['Adjusted_Volatility'] < train['Dynamic_Threshold']), 'Signal'] = -1

# Get GARCH volatility on the testing set
test['GARCH_Volatility'] = results.conditional_volatility

# Introduce volatility adjustments on the testing set
test['Adjusted_Volatility'] = volatility_multiplier * test['GARCH_Volatility']

# Calculate the Dynamic_Threshold for the testing set
test['Dynamic_Threshold'] = threshold_multiplier * test['Adjusted_Volatility'].rolling(window=20).mean()

# Implement mean reversion trading strategy with dynamic threshold on testing set
test['Signal'] = 0
test.loc[(test['Log_Returns'] < 0) & (test['Adjusted_Volatility'] < test['Dynamic_Threshold']), 'Signal'] = 1
test.loc[(test['Log_Returns'] > 0) & (test['Adjusted_Volatility'] < test['Dynamic_Threshold']), 'Signal'] = -1

# Calculate daily returns
train['Daily_Return'] = train['Close'].pct_change() * train['Signal'].shift(1)
test['Daily_Return'] = test['Close'].pct_change() * test['Signal'].shift(1)

# Cumulative returns
train['Cumulative_Return'] = (1 + train['Daily_Return']).cumprod()
test['Cumulative_Return'] = (1 + test['Daily_Return']).cumprod()

# Additional performance metrics
sharpe_ratio = np.sqrt(252) * np.mean(train['Daily_Return']) / np.std(train['Daily_Return'])
max_drawdown = np.min(train['Cumulative_Return'] - train['Cumulative_Return'].cummax())
cumulative_return = train['Cumulative_Return'].iloc[-1]

# Print the additional metrics and Hurst exponent
print(f'Hurst Exponent: {hurst_exponent}')
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Maximum Drawdown: {max_drawdown}')
print(f'Cumulative Return: {cumulative_return}')

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Close'], label='Train - Close Price')
plt.plot(test.index, test['Close'], label='Test - Close Price')
plt.scatter(train.index[train['Signal'] == 1], train['Close'][train['Signal'] == 1], marker='^', color='g', label='Buy Signal (Train)')
plt.scatter(train.index[train['Signal'] == -1], train['Close'][train['Signal'] == -1], marker='v', color='r', label='Sell Signal (Train)')
plt.scatter(test.index[test['Signal'] == 1], test['Close'][test['Signal'] == 1], marker='^', color='b', label='Buy Signal (Test)')
plt.scatter(test.index[test['Signal'] == -1], test['Close'][test['Signal'] == -1], marker='v', color='orange', label='Sell Signal (Test)')
plt.title('Mean Reversion Trading Strategy with Dynamic Threshold')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

