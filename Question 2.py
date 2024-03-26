
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import skew

# Load data from Excel file
file_path = 'D:/NEOMA/Python/EURUSD.xlsx'
df = pd.read_excel(file_path, parse_dates=['Date'], index_col='Date')

# Calculate daily returns
df['Returns'] = df['Close'].pct_change().dropna()

# Lag the Returns column to get X_t-1
df['Lagged_Returns'] = df['Returns'].shift(1)

# Parameters for the trading strategy
estimation_window = 80
critical_value = -1

# Initialize position and performance columns
df['Position'] = 0  # 0: No position, 1: Buy Euro, -1: Buy USD
df['Strategy_Returns'] = 0

# Initialize lists to store model parameters and cumulative returns
alpha_hat_list = []
beta_hat_list = []
t_stat_list = []
cumulative_returns_list = []

# Initial wealth
initial_wealth = 1
wealth_list = [initial_wealth]

# Loop through the data to implement the trading strategy
for i in range(estimation_window, len(df)):
    # Extract data for the estimation window
    window_data = df.iloc[i - estimation_window:i]

    # Drop rows with missing or infinite values
    window_data = window_data.replace([np.inf, -np.inf], np.nan).dropna()

    # Perform regression for the estimation window if there's enough data
    if len(window_data) > 2:  # Adjust this threshold as needed
        X = sm.add_constant(window_data['Lagged_Returns'])
        model = sm.OLS(window_data['Returns'], X)
        results = model.fit()

        # Extract coefficients and t-statistic for beta
        alpha_hat, beta_hat = results.params
        t_stat_beta = results.tvalues['Lagged_Returns']

        # Append values to lists
        alpha_hat_list.append(alpha_hat)
        beta_hat_list.append(beta_hat)
        t_stat_list.append(t_stat_beta)

        # Check trading conditions
        if alpha_hat > 0 and beta_hat < 0 and t_stat_beta < critical_value:
            # Compute expected change
            expected_change = alpha_hat + beta_hat * df['Close'].iloc[i]

            # Update position based on expected change
            if expected_change > 0:
                df.at[df.index[i], 'Position'] = 1  # Buy Euro
            elif expected_change < 0:
                df.at[df.index[i], 'Position'] = -1  # Buy USD

    # Calculate daily returns of the strategy
    df.at[df.index[i], 'Strategy_Returns'] = df['Position'].iloc[i] * df['Returns'].iloc[i]

    # Update wealth based on daily returns
    wealth_list.append(wealth_list[-1] * (1 + df['Strategy_Returns'].iloc[i]))

# Calculate Sharpe Ratio
sharpe_ratio = (np.mean(df['Strategy_Returns']) / np.std(df['Strategy_Returns'])) * np.sqrt(252)

# Calculate Maximum Drawdown
cumulative_returns = (1 + df['Strategy_Returns']).cumprod()
max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)

# Calculate Calmar Ratio
calmar_ratio = sharpe_ratio / max_drawdown

# Calculate the cumulative return of the strategy
cumulative_strategy_return = (1 + df['Strategy_Returns']).cumprod()

# Visualize the growth of the initial investment and cumulative return
plt.figure(figsize=(12, 6))
plt.plot(df.index[estimation_window:], wealth_list[1:], label='Strategy Wealth')
plt.plot(df.index[estimation_window:], cumulative_strategy_return[estimation_window:], label='Cumulative Strategy Return', linestyle='--')
plt.title('Growth of Initial Investment and Cumulative Strategy Return Over Time')
plt.xlabel('Date')
plt.ylabel('Wealth / Cumulative Return')
plt.legend()
plt.show()

# Print the cumulative return
final_cumulative_return = cumulative_strategy_return.iloc[-1]
print(f'Final Cumulative Return: {final_cumulative_return:.4f}')

# Print performance metrics
print('Mean Reversion Model Parameters:')
print(f'Estimated alpha (𝛼̂): {np.mean(alpha_hat_list):.4f}')
print(f'Estimated beta (𝛽̂): {np.mean(beta_hat_list):.4f}')
print(f'Mean t-statistic for beta: {np.mean(t_stat_list):.4f}')

# Calculate skewness of the strategy returns
strategy_skewness = skew(df['Strategy_Returns'].dropna())
print(f'Skewness of Strategy Returns: {strategy_skewness:.4f}')

# Calculate Standard Deviation and Downside Deviation
strategy_std_dev = df['Strategy_Returns'].std()
downside_returns = df['Strategy_Returns'][df['Strategy_Returns'] < 0]
downside_deviation = downside_returns.std()
print(f'Standard Deviation of Strategy Returns: {strategy_std_dev:.4f}')
print(f'Downside Deviation of Strategy Returns: {downside_deviation:.4f}')

# Print performance metrics
print(f'Sharpe Ratio: {sharpe_ratio:.4f}')
print(f'Maximum Drawdown: {max_drawdown:.4f}')
print(f'Calmar Ratio: {calmar_ratio:.4f}')
