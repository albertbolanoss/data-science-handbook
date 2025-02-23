import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


# Download and prepare the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# # Visualize the data
lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
# La función plt.axis se utiliza para configurar los límites de los ejes en el gráfico. 
#Esto permite controlar la región visible del gráfico para que se centre en el rango de datos de interés. Además, si se llama sin argumentos, plt.axis() devuelve los límites actuales de los ejes.
plt.axis([23_500, 62_500, 4, 9])
plt.show()
plt.savefig('output/life_satisfaction_vs_gpd.png')

# Select a linear model
# KNeighborsRegressor is an instance-based machine learning algorithm. This means that it does not learn an explicit model from the training data, 
# but stores the training instances and makes predictions based on the similarity between the new instances and the stored instances.
model = KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
print(model.predict(X_new)) # output: [[6.30165767]]