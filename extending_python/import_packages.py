
# pandas 
import pandas as pd

# Dataset adapted from here https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
data = pd.read_csv('car_data.csv')
print(data)# Code to show only the cars with a price >= 10000
print(data[data["Price"]>=10000])
# Show all the cars from the 2015 
print(data[data["Year"]==2015])
filtered_data = data[data["Year"]==2015]
print(filtered_data["Price"].median())


# matplotlib 
import matplotlib.pyplot as plt
plt.scatter(data["Kilometer"], data["Price"], color='red')
plt.title('Car Price vs. Kilometers Driven', fontsize=16)
plt.xlabel('Kilometers Driven')
plt.ylabel('Price (in USD)')
# Add the grid
plt.grid(True)
# Display the plot
plt.show()

