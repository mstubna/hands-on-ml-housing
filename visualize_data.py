import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

plt.close('all')

housing = pd.read_csv('data/housing.csv')
print(housing.head())
print(housing.info())
housing.hist(bins=50, figsize=(20, 15))

# create income categorical variable
housing['income_cat'] = pd.cut(
  housing['median_income'],
  bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
  labels=[1, 2, 3, 4, 5]
)

# create training and test data sets, stratifying on income category
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]

# show that the training data set has similar income distribution as full data set
fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
housing['income_cat'].hist(ax=axs[0])
axs[0].set_ylabel('Count')
axs[0].set_xlabel('Income category')
axs[0].set_title('Full data set')
strat_train_set['income_cat'].hist(ax=axs[1])
axs[1].set_xlabel('Income category')
axs[1].set_title('Training data set')

# copy training set
housing = strat_train_set.copy()
strat_train_set = strat_train_set.drop('income_cat', axis=1)
strat_test_set = strat_test_set.drop('income_cat', axis=1)

# scatterplot
cmap = plt.get_cmap('jet')
plt.figure(figsize=(10, 7), dpi=100)
plt.scatter(
  x=housing['longitude'],
  y=housing['latitude'],
  color='None',
  edgecolors=cmap(housing['median_house_value']/500000),
  s=housing['population']/60,
  linewidths=0.25,
  label='population'
)
plt.title('Population and house value by location')
plt.legend()

# test correlations
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']
corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

# construct data pipeline
housing_labels = strat_train_set['median_house_value'].copy()
housing = strat_train_set.drop('median_house_value', axis=1)
housing_num = housing.drop('ocean_proximity', axis=1)
housing_cat = housing[['ocean_proximity']]







plt.show()


