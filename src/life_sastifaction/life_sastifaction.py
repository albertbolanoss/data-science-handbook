import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path

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
