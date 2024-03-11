import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

# Load dataset
dataset = pd.read_excel(r"C:\Users\vijay\Documents\sem4\ML\lab3\irctc.xlsx")

# Display first few rows of dataset
a = dataset.head()
print(a)

# Calculate mean and variance of 'Price' column
mean = statistics.mean(dataset['Price'])
variance = statistics.variance(dataset['Price'])

# Check if 'Wednesday' data exists
wednesday_prices = []  # Initialize variable
if 'Wednesday' in dataset['Day'].values:
    wednesday_prices = dataset[dataset['Day'] == 'Wednesday']['Price']
    wednesday_mean = statistics.mean(wednesday_prices)
    print("Mean price for Wednesdays:", wednesday_mean)
    
    # Calculate probability of making a profit on Wednesday
    wednesday_profit_probability = len(dataset[(dataset['Day'] == 'Wednesday') & (dataset['Chg%'] > 0)]) / len(wednesday_prices)
    print("Probability of making a profit on Wednesday:", wednesday_profit_probability)
    
    # Calculate conditional probability of making profit, given that today is Wednesday
    conditional_probability = wednesday_profit_probability / (len(dataset[dataset['Day'] == 'Wednesday']) / len(dataset))
    print("Conditional probability of making profit, given that today is Wednesday:", conditional_probability)
else:
    print("No data available for Wednesdays.")

# Check if 'Apr' data exists
if 'Apr' in dataset['Month'].values:
    april_prices = dataset[dataset['Month'] == 'Apr']['Price']
    april_mean = statistics.mean(april_prices)
    print("Mean price for April:", april_mean)
else:
    print("No data available for April.")

# Calculate probability of making a loss
loss_probability = len(dataset[dataset['Chg%'] < 0]) / len(dataset['Chg%'])
print("Probability of making a loss over the stock:", loss_probability)

# Plot Chg% vs Day of the Week
plt.scatter(dataset['Day'], dataset['Chg%'])
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Chg% vs Day of the Week')
plt.show()
