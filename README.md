# 🚗 Car price prediction
This project focuses on predicting car prices using two machine learning models: Lasso Regression and Decision Tree Regression. The goal of this project is to provide an accurate estimate of car prices based on various features and specifications.

## Dataset
You can download this dataset in [**kaggle**](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction).

The dataset used for this project contains information about different cars, including variables such as `Name`, `Location`, `Year`, `Fuel_Type`,	`Engine`, `Seats` and other relevant features. It also includes the corresponding price of each car. This dataset is used for training and evaluating machine learning models.

## Installation and Setup
Clone the repository: git clone https://github.com/username/repo.git

Navigate to the project directory: ```cd car-price-prediction/model```

Install the required libraries: ```pip install pandas```,  ```pip install scikit-learn==0.22.2```

Run the Jupyter Notebook: **jupyter notebook car_price_prediction.ipynb**

## Usage
1. Load the dataset using ```pd.read_csv()``` function.
2. Clean the data and handle missing values.
3. Split the dataset into training and testing sets using ```train_test_split()``` function.
4. Create an instance of the Lasso Regression model: ```Lasso(alpha=1.0)``` and find the optimal alpha coefficient that minimizes the MAE (Mean Absolute Error).
5. Train the Lasso Regression model using the training set: `model_lasso.fit(X_train, y_train)`.
6. Evaluate the Lasso Regression model using the testing set: `model_lasso.score(X_test, y_test)`.
7. Create an instance of the Decision Tree Regression model: `DecisionTreeRegressor()` and find the tree size(`max_leaf_nodes`) that results in the smallest MAE.
8. Train the Decision Tree Regression model using the training set: `model_DTR.fit(X_train, y_train)`.
9. Evaluate the Decision Tree Regression model using the testing set: `model_DTR.score(X_test, y_test)`.
10. Predict car prices using both models: `lasso_model.predict(X_test) and model_DTR.predict(X_test)`.

## Model
### Lasso Regression Model
Lasso Regression is a linear regression model that performs both feature selection and regularization. It helps in reducing the complexity of the model by shrinking the coefficient estimates towards zero. The Lasso Regression model is utilized for predicting car prices based on the dataset.


### Decision Tree Regression Model
Decision Tree Regression is a non-linear regression model that utilizes a recursive partitioning method. It breaks down a dataset into smaller subsets based on different features and creates a tree-like model of decisions. The Decision Tree Regression model is used to predict car prices based on the dataset.


## Results
The performance of both models can be evaluated using metric such as Mean Absolute Error (`MAE`), which provide insights into how accurate the predictions are compared to the actual prices of the cars.


- Below are the obtained scores for each model 

    | | Model | Training Set Accuracy | Testing Set Accuracy |
    |-------- | -------- | -------- | -------- |
    0 | Lassso Regression     | 0.876661     | 0.844444     |
    1 | Decision Tree Regressor     | 0.968327     | 0.811434     |
    ---

## Conclusion
In this project, we utilized *Lasso Regression* and *Decision Tree Regression* models to predict the prices of cars based on various features. By training and evaluating these models on a dataset, we obtained predictions with a certain level of accuracy. 
