import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


# Get Data from https://github.com/ageron/data/raw/main/housing.tgz
def get_data(dataset):
    data = pd.read_csv(Path(dataset))

    # Getting information from dataset
    # Here you can check the total of instances (samples) and the type of each attribute
    data.info()

    # Also you an check the Non-Null count of each attribute (Note that there are some missing values for total_bedrooms)
    print("The Null Values {}".format(data.isnull().sum()))
    print("The Occean Proximity Values {}".format(data["ocean_proximity"].value_counts()))

    # Decritive statistics of the dataset
    # using tabulate to print the table in a fancy way
    # The count, mean, min, and max rows are self-explanatory. Note that the null values are ignored (so, for example, the count of total_bedrooms is 20,433, not 20,640). 
    # The std row shows the standard deviation, which measures how dispersed the values are.⁠5 
    # The 25%, 50%, and 75% rows show the corresponding percentiles: a percentile indicates the value below which a given percentage of observations in a group of observations fall. 
    # For example, 25% of the districts have a housing_median_age lower than 18, while 50% are lower than 29 and 75% are lower than 37. 
    # These are often called the 25th percentile (or first quartile), the median, and the 75th percentile (or third quartile).
    print(tabulate(data.describe(), headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))

    # Other way to get the statistics of the dataset is to plot a histogram for each numerical attribute
    data.hist(bins=50, figsize=(12, 8))
    plt.tight_layout() # Adjust the subplots to give more space between them
    plt.savefig("output/housing/describe_features.png")
    plt.close() 

    return data


# Load the dataset and analize it
housing = get_data(dataset="datasets/housing/housing.csv")

# Split the dataset into train and test sets
# Just Split the data into two sets: 80% for training and 20% for testing using a random seed to avoid different results each time you run the code
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# Create a new category attribute from median_income (numerical attribute)
# the bins to create are: from 0 to 1.5, from 1.5 to 3.0, from 3.0 to 4.5, from 4.5 to 6.0, and from 6.0 to infinity
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.savefig("output/housing/income_cat.png")
plt.close()

# Now you are ready to do stratified sampling based on the income category. 
# Scikit-Learn provides a number of splitter classes in the sklearn.model_selection package that implement various strategies 
# to split your dataset into a training set and a test set. Each splitter has a split() method that returns an iterator over 
# different training/test splits of the same data.

# Getting a unique set of training and testing using the income category as stratified 
# Here you can check that the proportion of each category is the same in the test and training sets and also mach the original dataset so this way avoid sample bias.
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
print("Test Income cat: {}".format(strat_test_set["income_cat"].value_counts() / len(strat_test_set)))
print("Training Income cat: {}".format(strat_train_set["income_cat"].value_counts() / len(strat_train_set)))
print("Dataset Income cat: {}".format(housing["income_cat"].value_counts() / len(housing)))


# This one split the dataset into 10 splits and each split has 80% for training and 20% for testing using the income category as stratified
# also ensure that the proportion of each category is the same in the test and training sets and also mach the original dataset so this way avoid sample bias.
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])


# Deleting the income_cat feature from the dataset because you don't need it anymore (just for get stratified sampling)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# Explore and visualize the data to gain insights
housing = strat_train_set.copy()

# Let start plotting the longitude and latitude to see the areas where the houses are located and its high-density areas
# The parameter alpha is used to show the high-density areas (allow to see other point that are overlapped)
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.savefig("output/housing/longitude_vs_latitud.png")
plt.close()

# Next, you look at the housing prices (Figure 2-13). 
# The radius of each circle represents the district’s population (option s), 
# and the color represents the price (option c). 
# Here you use a predefined color map (option cmap) called jet, which ranges from blue (low values) to red (high prices):⁠8
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.savefig("output/housing/housing_prices_and_rest.png")
plt.close()


# El Standard Correlation Coefficient, también conocido como Coeficiente de Correlación de Pearson, mide la relación lineal entre dos variables. Su valor está entre -1 y 1, donde:
# Since the dataset is not too large, you can easily compute the standard correlation coefficient (also called Pearson’s r) between every pair of attributes using the corr() method:
# The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation; 
# for example, the median house value tends to go up when the median income goes up. When the coefficient is close to –1, 
# it means that there is a strong negative correlation; you can see a small negative correlation between the latitude and the median house value 
# (i.e., prices have a slight tendency to go down when you go north). Finally, coefficients close to 0 mean that there is no linear correlation.
corr_matrix = housing.select_dtypes(include=[np.number]).corr()
print("Correlation median_house_value vs other features:\n{}".format(corr_matrix["median_house_value"].sort_values(ascending=False)))