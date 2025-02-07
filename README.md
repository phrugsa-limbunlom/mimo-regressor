# Feature-Driven Optimization of Syngas Production in Biomass Gasifiers using Machine Learning

<p align="center">
Feature-Driven Optimization of Syngas Production in Biomass Gasifiers using Machine Learning <a href="https://github.com/phrugsa-limbunlom/CE880_Case_Study_MIMO_Biosyngas_Prediction/blob/main/%5B2311569%5D_CE880_Case_Study.pdf">(Paper)</a>
  <br>
Author: Phrugsa Limbunlom (CSEE, University of Essex, 2023) 
</p>
<br>

<div align="center">
  <img alt="Pandas" src="https://img.shields.io/badge/-Pandas-4CAF50?style=flat&logo=Pandas&logoColor=white">
  <img alt="Numpy" src="https://img.shields.io/badge/-Numpy-013243?style=flat&logo=Numpy&logoColor=white">
  <img alt="Scikit-learn" src="https://img.shields.io/badge/-Scikit%20learn-orange?style=flat&logo=Scikit&logoColor=white">
</div>

## Pearson Correlation
``` python
correlation_matrix = df.corr()
plt.figure(figsize=(15,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Correlation Matrix of Input and Output variables')
plt.show()
plt.savefig("heatmap.png")
```
![image](https://github.com/user-attachments/assets/e4465c15-17cc-4738-b68c-7f3480f33bed)

## Model Training and Evaluation

1. Update the folder and path name of the data set
``` python
path = "/content/drive/MyDrive/Colab Notebooks/Data Science/Data Sci Case Study/"
file_name = "dataBiomass_CE880.xlsx"
```

2. Run the model to finetune hyperparameters
``` python
def decisiontree_regressor():

  regressor = DecisionTreeRegressor(random_state=42)

  feature_selector = RFE(regressor)

  pipe = Pipeline(steps=[('scaler', scaler), ('feature_selection', feature_selector), ('regressor', regressor)])

  param_grid = {
      'feature_selection__n_features_to_select' : [1,2,3,4,5,6,7,8],
      'regressor__max_depth': [None, 5, 10, 15],
      'regressor__min_samples_split': [2, 5, 10],
      'regressor__min_samples_leaf': [1, 2, 4]
  }

  search_dt_regressor = GridSearchCV(pipe, param_grid=param_grid, cv=4, n_jobs=-1)

  search_dt_regressor = search_dt_regressor.fit(X_train, y_train)

  score = rmse_cv(search_dt_regressor)
  score_r2 = r2_cv(search_dt_regressor)

  max_depth =  search_dt_regressor.best_params_["regressor__max_depth"]
  min_samples_split =  search_dt_regressor.best_params_["regressor__min_samples_split"]
  min_samples_leaf =  search_dt_regressor.best_params_["regressor__min_samples_leaf"]

  features =  search_dt_regressor.best_params_["feature_selection__n_features_to_select"]

  best_features_indices = search_dt_regressor.best_estimator_.named_steps['feature_selection'].get_support(indices=True)

  print(f"Best number of features : {features}")
  print(f"Features indices : {best_features_indices}")

  print(f"Best parameters: max_depth = {max_depth}, min_samples_split = {min_samples_split}, min_samples_leaf = {min_samples_leaf}")

  print("\nDecision tree rmse score: {:.3f} ({:.3f})\n".format(score.mean(), score.std()))
  print("\nDecision tree r2 score: {:.3f} ({:.3f})\n".format(score_r2.mean(), score_r2.std()))

  rmse = score.mean()
  r2 = score_r2.mean()

  return search_dt_regressor,round(rmse,3),round(r2,3)
```
3. Verify the model score by running unit test
``` python
search_dt_regressor, rmse, r2 = decisiontree_regressor()
assert math.isclose(rmse, 7.524)
assert math.isclose(r2, 0.705)
```
4. Run function to train the model
``` python
def train(model, X_train):

  scaler_x = scaler.fit(X_train)
  X_train_scaled = scaler_x.transform(X_train)
  y_train_scaled = scaler_y.fit_transform(y_train)

  model.fit(X_train_scaled,y_train_scaled)

  y_pred_scaled = model.predict(X_train_scaled)

  y_pred = scaler_y.inverse_transform(y_pred_scaled)

  train_loss = np.sqrt(mean_squared_error(y_train,y_pred))

  print("Train loss: {:.3f}\n".format(train_loss))

  return model, scaler_x, scaler_y
```

5. Run function to predict data
``` python
def predict(model, scaler_x, scaler_y, X_test):

  X_test_scaled = scaler_x.transform(X_test)

  y_pred_scaled = model.predict(X_test_scaled)

  y_pred = scaler_y.inverse_transform(y_pred_scaled)

  r2_arr = []
  rmse_arr = []

  for idx, col in enumerate(columns_target):
    r2 = r2_score(y_test[:,idx],y_pred[:,idx],force_finite=False)

    r2_arr.append(round(r2,3))

    rmse = np.sqrt(mean_squared_error(y_test[:,idx],y_pred[:,idx]))

    rmse_arr.append(round(rmse,3))

    print("\n[{}] r2 score: {:.3f}\n".format(col, r2))

    print("\n[{}] rmse score: {:.3f}\n".format(col, rmse))

  return np.array(r2_arr), np.array(rmse_arr)
```

6. Verify the scores of predicted data by runnning unit test
 
``` python
def test_prediction(r2,rmse,r2_expected,rmse_expected):
  np.testing.assert_array_equal(r2,r2_expected)
  np.testing.assert_array_equal(rmse,rmse_expected)
```

``` python
model,scaler_x, scaler_y = train(search_dt_regressor,X_train[:,[1,7]])
r2, rmse = predict(model,scaler_x, scaler_y,X_test[:,[1,7]])
r2_expected = np.array([0.838, 0.686, 0.646, 0.840, 0.685, 0.443, 0.242])
rmse_expected = np.array([4.389, 4.187, 5.387, 1.615, 1.502, 7.927, 17.554])
test_prediction(r2,rmse,r2_expected,rmse_expected)
```



