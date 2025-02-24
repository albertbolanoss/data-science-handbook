import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import pipeline

# Loading datasets:
datapath = Path("datasets/lifesat")

oecd_bli = pd.read_csv(datapath / "oecd_bli.csv")
gdp_per_capita = pd.read_csv(datapath / "gdp_per_capita.csv")

# Preprocess the GDP per capita data to keep only the year 2020:
gdp_year = 2020
gdppc_col = "GDP per capita (USD)"
lifesat_col = "Life satisfaction"
gdp_per_capita = gdp_per_capita[gdp_per_capita["Year"] == gdp_year]     # Keep only 2020 data
gdp_per_capita = gdp_per_capita.drop(["Code", "Year"], axis=1)          # Drop unnecessary columns Code and Year
gdp_per_capita.columns = ["Country", gdppc_col]                         # Rename original dataset columns names (the first two columns) to "Country" and "GDP per capita (USD)"
gdp_per_capita.set_index("Country", inplace=True)                       # Set the index of the dataset to be the "Country" instead of the default index (numeric)
                                                                        # inplace=True, means that the changes are made directly to the dataset and not is necessary to assign the result to a new variable
print(gdp_per_capita.head())


# Preprocess the OECD Better Life Index (BLI) data:
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]                              # Keep only the total values (not specific values for   
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value") # It's like groupby country (rows) and indicator (columns) and set the values for these specified in the values parameter (in this case de column also named Value)
# Example (not real data):
# INDICATOR   Air pollution  Educational attainment  Water quality
# Country                                                         
# Australia              5.0                   81.0          93.0
# Austria               16.0                   85.0          92.0
# Belgium               15.0                   77.0          84.0
print(oecd_bli.head())


# Merge the two datasets:
full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)   # Merge the two datasets by the index (Country)
print(full_country_stats.columns)                                                                       # show only the columns names
full_country_stats.sort_values(by=gdppc_col, inplace=True)                                              # Sort the dataset by the GDP per capita column
full_country_stats = full_country_stats[[gdppc_col, lifesat_col]]                                       # Keep only the GDP per capita and Life satisfaction columns

print(full_country_stats.head())

# To illustrate the risk of overfitting, I use only part of the data in most figures (all countries with a GDP per capita between min_gdp and max_gdp).
#  Later in the chapter I reveal the missing countries, and show that they don't follow the same linear trend at all.

min_gdp = 23_500
max_gdp = 62_500

country_stats = full_country_stats[(full_country_stats[gdppc_col] >= min_gdp) &
                                   (full_country_stats[gdppc_col] <= max_gdp)]
print(country_stats.head())


# Save the new datasets to csv files:
country_stats.to_csv(datapath / "lifesat.csv")
full_country_stats.to_csv(datapath / "lifesat_full.csv")

# Plot the data:
country_stats.plot(kind='scatter', figsize=(5, 3), grid=True, x=gdppc_col, y=lifesat_col)
#plt.show()
plt.savefig('output/life_satisfaction.png')

min_life_sat = 4
max_life_sat = 9

position_text = {
    "Turkey": (29_500, 4.2),
    "Hungary": (28_000, 6.9),
    "France": (40_000, 5),
    "New Zealand": (28_000, 8.2),
    "Australia": (50_000, 5.5),
    "United States": (59_000, 5.3),
    "Denmark": (46_000, 8.5)
}

for country, pos_text in position_text.items():
    pos_data_x = country_stats[gdppc_col].loc[country]
    pos_data_y = country_stats[lifesat_col].loc[country]
    country = "U.S." if country == "United States" else country
    plt.annotate(country, xy=(pos_data_x, pos_data_y),
                 xytext=pos_text, fontsize=12,
                 arrowprops=dict(facecolor='black', width=0.5,
                                 shrink=0.08, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "ro")

plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])

# plt.show()
plt.savefig('output/money_happy_scatterplot.png')


# Choses the  position_text countries to highlight in the plot:
highlighted_countries = country_stats.loc[list(position_text.keys())]
highlighted_countries[[gdppc_col, lifesat_col]].sort_values(by=gdppc_col)


# Using different parameters for the regression.
country_stats.plot(kind='scatter', figsize=(5, 3), grid=True,
                   x=gdppc_col, y=lifesat_col)

X = np.linspace(min_gdp, max_gdp, 1000)                         # Create a range of values from min_gdp to max_gdp with 1000 values

w1, w2 = 4.2, 0
plt.plot(X, w1 + w2 * 1e-5 * X, "r")                            # Plot a line equiation with y = 4.2 and m = 0 (color red)
plt.text(40_000, 4.9, fr"$\theta_0 = {w1}$", color="r")         # Add a text in the plot with the value of theta_0
plt.text(40_000, 4.4, fr"$\theta_1 = {w2}$", color="r")         # Add a text in the plot with the value of theta_1

w1, w2 = 10, -9
plt.plot(X, w1 + w2 * 1e-5 * X, "g")                            # Plot a line equiation with y = 10 and m = -9 (color green)
plt.text(26_000, 8.5, fr"$\theta_0 = {w1}$", color="g")
plt.text(26_000, 8.0, fr"$\theta_1 = {w2} \times 10^{{-5}}$", color="g")

w1, w2 = 3, 8
plt.plot(X, w1 + w2 * 1e-5 * X, "b")                            # Plot a line equiation with y = 3 and m = 8 (color blue)
plt.text(48_000, 8.5, fr"$\theta_0 = {w1}$", color="b")
plt.text(48_000, 8.0, fr"$\theta_1 = {w2} \times 10^{{-5}}$", color="b")

plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])

plt.savefig('output/tweaking_model_params_plot')

# Train a linear regression model:

# The fit function receive a matrix of two dimension so we need to use the values function that return a numpy array of 2 dimensions 
X_sample = country_stats[[gdppc_col]].values
y_sample = country_stats[[lifesat_col]].values

lin1 = linear_model.LinearRegression()
lin1.fit(X_sample, y_sample)

t0, t1 = lin1.intercept_[0], lin1.coef_.ravel()[0]
print(f"θ0={t0:.2f}, θ1={t1:.2e}")


# Plot the best model for the current data (using found parameters):
country_stats.plot(kind='scatter', figsize=(5, 3), grid=True,
                   x=gdppc_col, y=lifesat_col)

X = np.linspace(min_gdp, max_gdp, 1000)
plt.plot(X, t0 + t1 * X, "b")

plt.text(max_gdp - 20_000, min_life_sat + 1.9,
         fr"$\theta_0 = {t0:.2f}$", color="b")
plt.text(max_gdp - 20_000, min_life_sat + 1.3,
         fr"$\theta_1 = {t1 * 1e5:.2f} \times 10^{{-5}}$", color="b")

plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])

plt.savefig('output/best_fit_model_plot')



# Predict a value for Cyprus:
cyprus_gdp_per_capita = gdp_per_capita[gdppc_col].loc["Cyprus"]
print(cyprus_gdp_per_capita)

cyprus_predicted_life_satisfaction = lin1.predict([[cyprus_gdp_per_capita]])[0, 0]
print(cyprus_predicted_life_satisfaction)

country_stats.plot(kind='scatter', figsize=(5, 3), grid=True,
                   x=gdppc_col, y=lifesat_col)

X = np.linspace(min_gdp, max_gdp, 1000)
plt.plot(X, t0 + t1 * X, "b")

plt.text(min_gdp + 22_000, max_life_sat - 1.1,
         fr"$\theta_0 = {t0:.2f}$", color="b")
plt.text(min_gdp + 22_000, max_life_sat - 0.6,
         fr"$\theta_1 = {t1 * 1e5:.2f} \times 10^{{-5}}$", color="b")

plt.plot([cyprus_gdp_per_capita, cyprus_gdp_per_capita],
         [min_life_sat, cyprus_predicted_life_satisfaction], "r--")                 # Mark the point (cyprus_gdp_per_capita, cyprus_predicted_life_satisfaction) using a red dashed line. 
plt.text(cyprus_gdp_per_capita + 1000, 5.0,
         fr"Prediction = {cyprus_predicted_life_satisfaction:.2f}", color="r")
plt.plot(cyprus_gdp_per_capita, cyprus_predicted_life_satisfaction, "ro")

plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])

plt.savefig('output/predict_cyprus_plot')


# Now we are going to operate with full_country_stats (the all countries dataset).

missing_data = full_country_stats[(full_country_stats[gdppc_col] < min_gdp) |
                                  (full_country_stats[gdppc_col] > max_gdp)]

position_text_missing_countries = {
    "South Africa": (20_000, 4.2),
    "Colombia": (6_000, 8.2),
    "Brazil": (18_000, 7.8),
    "Mexico": (24_000, 7.4),
    "Chile": (30_000, 7.0),
    "Norway": (51_000, 6.2),
    "Switzerland": (62_000, 5.7),
    "Ireland": (81_000, 5.2),
    "Luxembourg": (92_000, 4.7),
}

full_country_stats.plot(kind='scatter', figsize=(8, 3),
                        x=gdppc_col, y=lifesat_col, grid=True)

for country, pos_text in position_text_missing_countries.items():
    pos_data_x, pos_data_y = missing_data.loc[country]
    plt.annotate(country, xy=(pos_data_x, pos_data_y),
                 xytext=pos_text, fontsize=12,
                 arrowprops=dict(facecolor='black', width=0.5,
                                 shrink=0.08, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "rs")

# Draw the line using the parameters of the partial dataset:
X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0 + t1 * X, "b:")

lin_reg_full = linear_model.LinearRegression()
Xfull = np.c_[full_country_stats[gdppc_col]]
yfull = np.c_[full_country_stats[lifesat_col]]
lin_reg_full.fit(Xfull, yfull)

t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_.ravel()[0]
X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0full + t1full * X, "k")

plt.axis([0, 115_000, min_life_sat, max_life_sat])

plt.savefig('output/representative_training_data_scatterplot')



# Plop the full data with overfitting

full_country_stats.plot(kind='scatter', figsize=(8, 3),
                        x=gdppc_col, y=lifesat_col, grid=True)

poly = preprocessing.PolynomialFeatures(degree=10, include_bias=False)
scaler = preprocessing.StandardScaler()
lin_reg2 = linear_model.LinearRegression()

pipeline_reg = pipeline.Pipeline([
    ('poly', poly),
    ('scal', scaler),
    ('lin', lin_reg2)])
pipeline_reg.fit(Xfull, yfull)
curve = pipeline_reg.predict(X[:, np.newaxis])
plt.plot(X, curve)

plt.axis([0, 115_000, min_life_sat, max_life_sat])

plt.savefig('output/overfitting_model_plot')



# Este fragmento de código extrae y muestra la satisfacción de vida (lifesat_col) de los países cuyo nombre contiene la letra "W"
w_countries = [c for c in full_country_stats.index if "W" in c.upper()]
full_country_stats.loc[w_countries][lifesat_col]

all_w_countries = [c for c in gdp_per_capita.index if "W" in c.upper()]
gdp_per_capita.loc[all_w_countries].sort_values(by=gdppc_col)


country_stats.plot(kind='scatter', x=gdppc_col, y=lifesat_col, figsize=(8, 3))
missing_data.plot(kind='scatter', x=gdppc_col, y=lifesat_col,
                  marker="s", color="r", grid=True, ax=plt.gca())

country_stats.plot(kind='scatter', x=gdppc_col, y=lifesat_col, figsize=(8, 3))
missing_data.plot(kind='scatter', x=gdppc_col, y=lifesat_col,
                  marker="s", color="r", grid=True, ax=plt.gca())


# La regularización es una técnica que ayuda a evitar el sobreajuste (overfitting) en los modelos de aprendizaje automático, 
# agregando una penalización a los coeficientes del modelo.

# Cuando un modelo tiene overfitting, significa que está demasiado ajustado a los datos de entrenamiento y no generaliza bien a datos nuevos. 
# La regularización ayuda a reducir la complejidad del modelo, evitando que aprenda demasiado los detalles y el ruido del dataset.

X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0 + t1*X, "b:", label="Linear model on partial data")
plt.plot(X, t0full + t1full * X, "k-", label="Linear model on all data")

ridge = linear_model.Ridge(alpha=10**9.5)
X_sample = country_stats[[gdppc_col]]
y_sample = country_stats[[lifesat_col]]
ridge.fit(X_sample, y_sample)
t0ridge, t1ridge = ridge.intercept_[0], ridge.coef_.ravel()[0]
plt.plot(X, t0ridge + t1ridge * X, "b--",
         label="Regularized linear model on partial data")
plt.legend(loc="lower right")

plt.axis([0, 115_000, min_life_sat, max_life_sat])

plt.savefig("output/ridge_regression_plot")