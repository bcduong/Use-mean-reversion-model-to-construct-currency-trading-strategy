import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data from Excel file
file_path = 'D:/NEOMA/Python/EURUSD.xlsx'
df = pd.read_excel(file_path)

# Calculate ∆𝑋𝑡
df['∆𝑋𝑡'] = df['Close'].diff()

# Lagged variable X𝑡−1
df['X𝑡−1'] = df['Close'].shift(1)

# Drop NaN values
df = df.dropna()

# Perform time-series regression
X = sm.add_constant(df['X𝑡−1'])
y = df['∆𝑋𝑡']
model = sm.OLS(y, X)
results = model.fit()

# Extract coefficients
𝛼 = results.params['const']
𝛽 = results.params['X𝑡−1']
𝜃 = -𝛽  # Speed of mean reversion

# Calculate long-run mean 𝜇
𝜇 = -𝛼 / 𝛽

# Print results
print(f'𝛼: {𝛼}, 𝛽: {𝛽}, 𝜇: {𝜇}, 𝜃: {𝜃}')
print()

# Check if 𝛼 > 0 and 𝛽 < 0
if 𝛼 > 0 and 𝛽 < 0:
    print('The data exhibits mean reversion feature.')
else:
    print('The data does not exhibit mean reversion feature.')

# Extract t-statistic of the 𝛽 coefficient
t_statistic = results.tvalues['X𝑡−1']
critical_value = -1  # Set your critical value here

# Compare t-statistic with critical value
if t_statistic < critical_value:
    print(f'T-statistic ({t_statistic}) is statistically significant.')

# Visualize ∆𝑋𝑡 as the y-axis
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['∆𝑋𝑡'], label='∆XT')
plt.xlabel('Date')
plt.ylabel('∆Xt')
plt.title('∆Xt over Time')
plt.legend()
plt.show()

