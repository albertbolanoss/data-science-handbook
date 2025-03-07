import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix




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

### 1. Get the Data ###
# Load the dataset and analize it
housing = get_data(dataset="datasets/housing/housing.csv")


### 2. Split the dataset into train and test sets ###
# Just Split the data into two sets: 80% for training and 20% for testing using a random seed to avoid different results each time you run the code
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


## Checking the media income distribution to select to stratify the dataset in the next step
# Create a new category attribute from median_income (numerical attribute)
# Creates a new column called “income_cat” in the DataFrame housing, sorting the values of “median_income” into discrete categories using pd.cut().
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5]) 
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.savefig("output/housing/income_cat.png")
plt.close()


### 3. Split the dataset into train and test sets usin stratified ###
# Now you are ready to do stratified sampling based on the income category. 
# Scikit-Learn provides a number of splitter classes in the sklearn.model_selection package that implement various strategies 
# to split your dataset into a training set and a test set. Each splitter has a split() method that returns an iterator over 
# different training/test splits of the same data.

# Getting a unique set of training and testing using the income category as stratified 
# Here you can check that the proportion of each category is the same in the test and training sets and also mach the original dataset so this way avoid sample bias.

# Stratify is specify the column to be used for balancing the datasets with same impact in each set (train and test and validation).

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


### 4. Explore and visualize the data to gain insights ###
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


# El Standard Correlation Coefficient, también conocido como Coeficiente de Correlación de Pearson, mide la relación lineal entre dos variables. 
# Su valor está entre -1 y 1, donde:
# Since the dataset is not too large, you can easily compute the standard correlation coefficient (also called Pearson’s r) between every pair of attributes using the corr() method:
# The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation; 
# for example, the median house value tends to go up when the median income goes up. When the coefficient is close to –1, 
# it means that there is a strong negative correlation; you can see a small negative correlation between the latitude and the median house value 
# (i.e., prices have a slight tendency to go down when you go north). Finally, coefficients close to 0 mean that there is no linear correlation.
corr_matrix = housing.select_dtypes(include=[np.number]).corr()
print("Correlation median_house_value vs other features:\n{}".format(corr_matrix["median_house_value"].sort_values(ascending=False)))

# Plotting the correlation matrix
# Another way to check for correlation between attributes is to use the pandas scatter_matrix() function,
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.savefig("output/housing/scatter_matrix.png")
plt.close()

# Finding more usable features
# The numbers of rooms per house holds. 
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]             # mean total rooms per household
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]          # mean total bedrooms per total rooms
housing["people_per_house"] = housing["population"] / housing["households"]             # mean population per household

# Now let's check the correlation matrix again to see if the new features are more correlated with the median house value.
corr_matrix = housing.select_dtypes(include=[np.number]).corr()
print("New features correlation\n{}".format(corr_matrix["median_house_value"].sort_values(ascending=False)))


# Prepare the data for Machine Learning algorithms
# Split the features and labels
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


### 5. Cleaning the data  ### 

# The most of machine learning algorithms cannot work with missing features, so let's create a few functions to take care

# 1. Get rid of the corresponding districts.
# housing.dropna(subset=["total_bedrooms"], inplace=True)

# 2. Get rid of the whole attribute.
# housing.drop("total_bedrooms", axis=1)

# 3. Set the missing values to some value (zero, the mean, the median, etc.). This is called imputation.
# median = housing["total_bedrooms"].median()  # option 3
# housing["total_bedrooms"].fillna(median, inplace=True)

# Scikit-Learn provides a handy class to take care of missing values: SimpleImputer.
# the advance of use this one, it's that use with train and test and validation sets
# because it's store the median value of each attribute in the statistics_ instance variable
# also you can use these strategies: mean, median, most_frequent, constant, fill_value
# The last two strategies support non-numerical data.

# Fising Missing Values
imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
imputer.statistics_         # The imputer delivers the median value of each attribute

# also you can use fit_transform to do both operations, sometimes it is optimized and run much faster 

# Remplace the missing values with the median value
X = imputer.transform(housing_num)

# Scikit-Learn transformers output NumPy arrays (or sometimes SciPy sparse matrices)
# To convert again to pandas
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


# Now let's preprocess the categorical input feature, ocean_proximity:
housing_cat = housing[["ocean_proximity"]]

# Given that Machine learning algoritms works with numbers
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# To check the encoded categories you can get the list of categories using the categories_ instance variable. 
# It is a list containing a 1D array of categories for each categorical attribute (in this case, a list containing a single array since there is just one categorical attribute):
print("The Ordinal Categories: \n{}".format(ordinal_encoder.categories_))

# The issue is that the result of the order of categories can no be the expected.
# but it is obviously not the case for the ocean_proximity column (for example, categories 0 and 4 are clearly more similar than categories 0 and 1). 
# To fix this issue, a common solution is to create one binary attribute per category: one attribute equal to 1 when the category is "<1H OCEAN" (and 0 otherwise), 
# another attribute equal to 1 when the category is "INLAND" (and 0 otherwise), and so on. This is called one-hot encoding, because only one attribute will be equal to 1 (hot), 
# while the others will be 0 (cold). The new attributes are sometimes called dummy attributes. Scikit-Learn provides a OneHotEncoder class to convert categorical values into one-hot vectors:

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# Convert each category into a one-hot vector assign a 1 to the category and 0 to the others
# This is a sparse matrix, to convert to a dense matrix you can use toarray() method
print("The OneHot Categories: \n{}".format(housing_cat_1hot.toarray()))
#  Dense matrix                                                    Sparse matrix  
#  [0. 0. 0. 1. 0.]  # one-hot vector for category 0: "<1H OCEAN"  => 0, 3 = 1
#  [1. 0. 0. 0. 0.]  # one-hot vector for category 1: "INLAND"     => 1, 0 = 1 
#  [0. 1. 0. 0. 0.]  # one-hot vector for category 2: "ISLAND"     => 2, 1 = 1

# The categories are stored in the categories_ instance variable as a list of arrays containing a 1D array of categories for each categorical attribute:
print("Finaly the Categories: {}".format(cat_encoder.get_feature_names_out()))


# Additionally, Pandas has a function called get_dummies() that allow to apply to column(s) and convert to hot vector).
cat_encoder.handle_unknown = "ignore" # To avoid errors when new categories appear in the test set
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY", "<2H OCEAN"]})
print("Dummy : \n{}".format(cat_encoder.transform(df_test)))


### 6. Fixing Distribution with heavy tail #

## Before apply the Feacture scaling (next topic).
# When a feature’s distribution has a heavy tail (i.e., when values far from the mean are not exponentially rare),
# both min-max scaling and standardization will squash most values into a small range. 
# Machine learning models generally don’t like this at all
# So before scaling features, you should try the following:
# 1. For example, a common way to do this for positive features with a heavy tail to the right is to replace the feature with its square root (or raise the feature to a power between 0 and 1)
# 2. If the feature has a really long and heavy tail, such as a power law distribution, then replacing the feature with its logarithm may help
# the population feature roughly follows a power law: districts with 10,000 inhabitants are only 10 times less frequent than districts with 1,000 inhabitants, not exponentially less frequent. Figure 2-17 shows how much better this feature looks when you compute its log: it’s very close to a Gaussian distribution (i.e., bell-shaped).

# Showing the population distribution with high tail.

housing["population"].hist(bins=10, figsize=(12, 8))
plt.tight_layout() # Adjust the subplots to give more space between them
plt.xlabel("Population")
plt.ylabel("Number of districts")
plt.savefig("output/housing/population_heavy_distribution.png")
plt.close()

# Showing the population distribution with high tail unsing bin personalized with min and max values
min_val = housing['population'].min()
max_val = housing['population'].max()
bins = np.linspace(min_val, max_val, 10)

housing["population_count"] = pd.cut(housing["population"], bins=bins, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]) 
housing["population_count"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Population")
plt.ylabel("Number of districts")
plt.savefig("output/housing/population_heavy_distribution_bins.png")
plt.close()

# Applying the log to the population feature
housing["population_log"] = np.log(housing["population"])
housing["population_log"].hist(bins=10, figsize=(12, 8))
plt.xlabel("Population")
plt.ylabel("Number of districts")
plt.savefig("output/housing/population_distribution_log.png")
plt.close()

## Bucketizing the feature
# Another approach to handle heavy-tailed features consists in bucketizing the feature



# A percentile is a statistical measure that indicates the relative position of a value within a dataset. 
# The p-th percentile represents the value below which p% of the data falls. 
# For example, the 90th percentile means that 90% of the values are below it, and 10% are above it.
# Here we are going to find the quartiles (25th, 50th, 75th) and the 90th percentile of the median_income feature
num_buckets = 10
housing["income_bucket"] = pd.qcut(housing["median_income"], q=num_buckets, labels=False)
print("Income Buckets: {}".format(housing["income_bucket"].value_counts().sort_index()))
# Now we can normalize the income_bucket feature to a range of 0 to 1
housing["income_bucket_scaled"] = housing["income_bucket"] / (num_buckets - 1)

housing["income_bucket_scaled"].hist(bins=10, figsize=(12, 8))
plt.tight_layout() # Adjust the subplots to give more space between them
plt.xlabel("Population")
plt.ylabel("Number of districts")
plt.savefig("output/housing/income_bucket_scaled.png")
plt.close()


### 7. Feature Scaling Transformation ###

# Most of Machine learning algoritms don't perform so well when the input numerical attributes have a diferent scales
# Note that while the training set values will always be scaled to the specified range, if new data contains outliers, 
# these may end up scaled outside the range. If you want to avoid this, just set the clip hyperparameter to True.

# Min-max scaling (many people call this normalization) is the simplest: 
# for each attribute, the values are shifted and rescaled so that they end up ranging from 0 to 1. 
# This is performed by subtracting the min value and dividing by the difference between the min and the max. 
# Scikit-Learn provides a transformer called MinMaxScaler for this. 
# It has a feature_range hyperparameter that lets you change the range if, 
# for some reason, you don’t want 0–1 (e.g., neural networks work best with zero-mean inputs, so a range of –1 to 1 is preferable). It’s quite easy to use:

# Also named Normalization
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
print("The Min-Max Scaling: {}".format(housing_num_min_max_scaled))


## Standarization
# Standardization is different: first it subtracts the mean value (so standardized values have a zero mean), 
# then it divides the result by the standard deviation (so standardized values have a standard deviation equal to 1). 
# Unlike min-max scaling, standardization does not restrict values to a specific range. 
# However, standardization is much less affected by outliers. 
# For example, suppose a district has a median income equal to 100 (by mistake), instead of the usual 0–15. Min-max scaling to the 0–1 range would map this outlier down to 1 and it would crush all the other values down to 0–0.15, 
# whereas standardization would not be much affected. Scikit-Learn provides a transformer called StandardScaler for standardization:

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
print("The Standardization Scaling: {}".format(housing_num_std_scaled))

# Tip
# If you want to scale a sparse matrix without converting it to a dense matrix first, 
# you can use a StandardScaler with its with_mean hyperparameter set to False: 
# it will only divide the data by the standard deviation, without subtracting the mean (as this would break sparsity).

# With_mean=True (default):
# Subtracts the mean and divides by the standard deviation.
# Converts the matrix to dense, which can be problematic for large sparse matrices.
# With with_mean=False:
# Only divides the values by the standard deviation without subtracting the mean.
# Preserves sparse structure (useful for scipy.sparse matrices).

std_scaler_without_mean = StandardScaler(with_mean=False)
housing_num_sparse = csr_matrix(housing_num.values)
housing_num_std_scaled_without_mean = std_scaler_without_mean.fit_transform(housing_num_sparse)
print("The Standardization Scaling with out Mean: {}".format(housing_num_std_scaled_without_mean))

