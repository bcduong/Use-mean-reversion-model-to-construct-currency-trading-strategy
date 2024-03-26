

import os
import pandas as pd
import numpy as np
import arch
from sklearn.model_selection import train_test_split
from hurst import compute_Hc
import matplotlib.pyplot as plt

def mean_reversion_strategy(file_path):
    # Load data from Excel file
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

    return train, test

# Specify the directory containing the files
directory_path = 'D:/NEOMA/Python/'

# Initialize empty DataFrames to store results
combined_train_results = pd.DataFrame()
combined_test_results = pd.DataFrame()

# Iterate through files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(directory_path, filename)
        train_results, test_results = mean_reversion_strategy(file_path)

        # Combine results for each currency pair
        combined_train_results = pd.concat([combined_train_results, train_results['Cumulative_Return']], axis=1)
        combined_test_results = pd.concat([combined_test_results, test_results['Cumulative_Return']], axis=1)

# Calculate combined portfolio returns
combined_train_results['Portfolio_Return'] = combined_train_results.mean(axis=1)
combined_test_results['Portfolio_Return'] = combined_test_results.mean(axis=1)

# Additional performance metrics for the combined portfolio
sharpe_ratio_portfolio = np.sqrt(252) * np.mean(combined_train_results['Portfolio_Return']) / np.std(combined_train_results['Portfolio_Return'])
max_drawdown_portfolio = np.min(combined_train_results['Portfolio_Return'] - combined_train_results['Portfolio_Return'].cummax())
cumulative_return_portfolio = combined_train_results['Portfolio_Return'].iloc[-1]

# Print the additional metrics for the combined portfolio
print('\nCombined Portfolio Metrics:')
print(f'Sharpe Ratio: {sharpe_ratio_portfolio}')
print(f'Maximum Drawdown: {max_drawdown_portfolio}')
print(f'Cumulative Return: {cumulative_return_portfolio}')

# Visualize the results for the combined portfolio
plt.figure(figsize=(12, 6))
for col in combined_train_results.columns:
    plt.plot(combined_train_results.index, combined_train_results[col], label=f'Train - {col} Close Price')

plt.plot(combined_test_results.index, combined_test_results['Portfolio_Return'], label='Test - Portfolio Return', linewidth=2, color='black')
plt.title('Mean Reversion Trading Strategy with Dynamic Threshold - Combined Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
