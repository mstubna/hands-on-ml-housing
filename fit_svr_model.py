import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import expon, reciprocal
from sklearn.svm import SVR

n_jobs = -1
rooms_index, bedrooms_index, population_index, households_index, income_index = 1, 2, 3, 4, 5

#
# Combines existing attributes into new attributes
#
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
  def __init__(self, include_bedrooms_per_room=True):
    self.include_bedrooms_per_room = include_bedrooms_per_room

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    rooms_per_household = X[:, rooms_index] / X[:, households_index]
    population_per_household = X[:, population_index] / X[:, households_index]
    X = np.c_[X, rooms_per_household, population_per_household]
    if self.include_bedrooms_per_room:
      bedrooms_per_room = X[:, bedrooms_index] / X[:, rooms_index]
      X = np.c_[X, bedrooms_per_room]
    return X

#
# Create new "cluster" attribute
#
class LatLongClusterer(BaseEstimator, TransformerMixin):
  def __init__(self, n_clusters=100):
    self.n_clusters = n_clusters

  def fit(self, X, y=None):
    kmeans = KMeans(
      n_clusters=self.n_clusters,
      n_jobs=n_jobs
    )
    kmeans.fit(X)
    self.kmeans_ = kmeans
    return self

  def transform(self, X, y=None):
    clusters = self.kmeans_.predict(X)
    return clusters.reshape(-1, 1)

#
# Load the data
#

housing = pd.read_csv('data/housing.csv')

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
strat_train_set = strat_train_set.drop('income_cat', axis=1)
strat_test_set = strat_test_set.drop('income_cat', axis=1)

# copy training set
housing = strat_train_set.copy()

#
# construct data pipeline
#

housing_labels = strat_train_set['median_house_value'].copy()
housing = strat_train_set.drop('median_house_value', axis=1)
housing_num = housing.drop(['ocean_proximity', 'longitude', 'latitude'], axis=1)
housing_cat = housing[['ocean_proximity']]
num_attrs = list(housing_num)
lat_long_attrs = ['longitude', 'latitude', 'median_income']
categorical_attrs = ['ocean_proximity']

num_pipeline = Pipeline([
  ('median_imputer', SimpleImputer(strategy='median')),
  ('combined_attributes_adder', CombinedAttributesAdder()),
  ('std_scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
  ('categorize', OneHotEncoder(drop='first'))
])
lat_long_pipeline = Pipeline([
  ('lat_long_clusterer', LatLongClusterer(n_clusters=1000)),
  ('categorize', OneHotEncoder(categories='auto', drop='first'))
])
preparation_pipeline = ColumnTransformer([
  ('numbers', num_pipeline, num_attrs),
  ('categorical', categorical_pipeline, categorical_attrs),
  ('lat_long', lat_long_pipeline, lat_long_attrs)
])
pipeline = Pipeline([
  ('prepare', preparation_pipeline),
  ('model', SVR())
])

#
# Scatterplot of clusters
#

preparation_pipeline.fit_transform(housing)
cmap = plt.get_cmap('jet')
plt.figure(figsize=(10, 7), dpi=100)
plt.scatter(
  x=housing['longitude'],
  y=housing['latitude'],
  color='None',
  edgecolors=cmap(housing['median_income'].to_numpy()/10),
  # s=housing['population']/60,
  linewidths=0.25,
  label='population'
)
cluster_centers = preparation_pipeline.named_transformers_['lat_long'].named_steps['lat_long_clusterer'].kmeans_.cluster_centers_
plt.scatter(
  x=cluster_centers[:,0],
  y=cluster_centers[:,1],
  color=cmap(cluster_centers[:,2]/10),
  marker='o'
)
plt.title('Population and house value by location')
plt.legend()
plt.show()

#
# Fit the model pipeline
# 

# Support Vector Regressor
param_distributions = {
  'prepare__lat_long__lat_long_clusterer__n_clusters': [1000, 1500, 2000],
  'model__kernel': ['rbf'],
  'model__C': reciprocal(20, 200000),
  'model__gamma': expon(scale=1.0)
}
rnd_search = RandomizedSearchCV(
  estimator=pipeline,
  param_distributions=param_distributions,
  cv=5,
  scoring='neg_mean_squared_error',
  return_train_score=True,
  n_iter=30,
  verbose=2,
  n_jobs=n_jobs
)
rnd_search.fit(housing, housing_labels)
sv_reg = rnd_search.best_estimator_
housing_predictions = sv_reg.predict(housing)
sv_score = rnd_search.best_score_
sv_rmse_score = np.sqrt(-sv_score)

#
# display the results
#

print('SV Regression:\n')
print('Best params: ', rnd_search.best_params_)
print('Best score: ', sv_rmse_score)
print('\n\n')
