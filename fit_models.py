import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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
rooms_index, bedrooms_index, population_index, households_index = 3, 4, 5, 6

## Combines existing attributes into new attributes
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
  def __init__(self, include_bedrooms_per_room = True):
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

# display summary statistics about cross validation scores
def display_scores (scores):
  print('RMSE scores:\n')
  print('Mean:', scores.mean())
  print('Std:', scores.std())
  print('\n')

# display summary of the features and their importance
def display_features (features):
  print('Features:\n')
  extra_attrs = ['rooms_per_household', 'population_per_household', 'bedrooms_per_room']
  cat_encoder = full_pipeline.named_transformers_['categorical'].named_steps['categorize']
  cat_one_hot_attrs = list(cat_encoder.categories_[0])
  attrs = num_attrs + extra_attrs + cat_one_hot_attrs
  scores = sorted(zip(features, attrs), reverse=True, key=lambda obj: abs(obj[0]))
  for items in scores:
    print(items)
  print('\n')

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

# construct data pipeline
housing_labels = strat_train_set['median_house_value'].copy()
housing = strat_train_set.drop('median_house_value', axis=1)
housing_num = housing.drop('ocean_proximity', axis=1)
housing_cat = housing[['ocean_proximity']]
num_attrs = list(housing_num)
categorical_attrs = ['ocean_proximity']

num_pipeline = Pipeline([
  ('median_imputer', SimpleImputer(strategy='median')),
  ('combined_attributes_adder', CombinedAttributesAdder()),
  ('std_scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
  ('categorize', OneHotEncoder(drop='first'))
])
full_pipeline = ColumnTransformer([
  ('numbers', num_pipeline, num_attrs),
  ('categorical', categorical_pipeline, categorical_attrs)  
])

housing_prepared = full_pipeline.fit_transform(housing)

# Linear regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)
lin_scores = cross_val_score(
  lin_reg,
  housing_prepared,
  housing_labels,
  scoring='neg_mean_squared_error',
  cv=10,
  n_jobs=n_jobs
)
lin_rmse_scores = np.sqrt(-lin_scores)
print('Linear Regression:\n')
display_scores(lin_rmse_scores)
display_features(lin_reg.coef_ / housing_labels.mean())
print('\n\n')

# Random forest
param_grid = [
  { 'n_estimators': [3, 10, 30, 100, 150, 200], 'max_features': [2, 4, 6, 8] }
]
forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
  estimator=forest_reg,
  param_grid=param_grid,
  cv=5,
  scoring='neg_mean_squared_error',
  return_train_score=True,
  verbose=2,
  n_jobs=n_jobs
)
grid_search.fit(housing_prepared, housing_labels)
forest_reg = grid_search.best_estimator_
housing_predictions = forest_reg.predict(housing_prepared)
forest_scores = cross_val_score(
  forest_reg,
  housing_prepared,
  housing_labels,
  scoring='neg_mean_squared_error',
  cv=10,
  n_jobs=n_jobs
)
forest_rmse_scores = np.sqrt(-forest_scores)
print('Random Forest:\n')
print('Best params: ', grid_search.best_params_)
display_scores(forest_rmse_scores)
display_features(forest_reg.feature_importances_)
print('\n\n')

# Support Vector Regressor
param_distributions = {
  'kernel': ['linear', 'rbf'],
  'C': reciprocal(20, 200000),
  'gamma': expon(scale=1.0)
}
sv_reg = SVR()
rnd_search = RandomizedSearchCV(
  estimator=sv_reg,
  param_distributions=param_distributions,
  cv=5,
  scoring='neg_mean_squared_error',
  return_train_score=True,
  n_iter=50,
  verbose=2,
  n_jobs=n_jobs
)
rnd_search.fit(housing_prepared, housing_labels)
sv_reg = rnd_search.best_estimator_
housing_predictions = sv_reg.predict(housing_prepared)
sv_scores = cross_val_score(
  sv_reg,
  housing_prepared,
  housing_labels,
  scoring='neg_mean_squared_error',
  cv=10,
  n_jobs=n_jobs
)
sv_rmse_scores = np.sqrt(-sv_scores)
print('SV Regression:\n')
print('Best params: ', rnd_search.best_params_)
display_scores(sv_rmse_scores)
# display_features(sv_reg.coef_[0] / housing_labels.mean()) # only works for linear kernel
print('\n\n')
