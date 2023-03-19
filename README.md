# Predicting Airbnb Prices in New York City
This project aims to predict the prices of Airbnb listings in New York City based on various features using regression models. It uses a dataset of Airbnb listings in New York City, which includes information such as location, room type, number of reviews, availability, and host name. 

## Dataset
The dataset used in this project is available on <b><u><a>https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data</a></u></b>. It contains information on over 48,000 Airbnb listings in New York City.

## Approach
The approach used in this project involves several steps ranging from data preprocessing, exploratory data analysis, feature engineering, to model selection, training and evaluation. The following are the main steps involved:

<h3>Exploratory data analysis (EDA):</h3> We use visualization techniques to better understand the distribution and relationship of the dataset features, and to gain insights about the data. The EDA helps us to decide which features to use in the modeling process. The process is as follows:
1. We visualize different features (such as room_type, neighbourhood_group, and neighbourhood), and there relationship with the target variable (price).
<img src="https://user-images.githubusercontent.com/127759119/226188898-323b503b-a55e-405f-94f7-f425cab546cf.png" width="500">
We also analyse the relation between the variable themselves (such as Distribution of neighbourhood_group and room_type with Respect to Latitude and Longitude).
<img src="https://user-images.githubusercontent.com/127759119/226188732-592ab329-56c0-4253-af06-d453d0f03f97.png" width="500">
Next, we visualize the distribution of the target variable (price) and its log transformations.
<img src="https://user-images.githubusercontent.com/127759119/226188754-893484a7-2c7d-45bc-b782-37eab9ab771d.png" width="500">
We, also visualise the correlation among different numerical features of the dataset using seaborn heatmap.
<img src="https://user-images.githubusercontent.com/127759119/226188807-abf3acda-a3d0-451e-9d82-ee2c3132812d.png" width="500">
<h3>Data cleaning and preparation:</h3>
 The dataset contains missing values and some outliers, which need to be addressed. 
1. We detect the outliers  and remove these in target variable (price).
<img src="https://user-images.githubusercontent.com/127759119/226188780-5c8db7be-c786-45b7-82d9-005140a6492e.png" width="500">
  
2. Next, we we fill the missing values in reviews_per_month, drop irrelevant columns, and apply the log transfoirmation to target variable (price).
3. Finally, we split the dataset into train and test with 1:4 ratio. 

  <h3>Feature engineering: </h3>
  We create new features from the existing ones to help the machine learning algorithms better capture the patterns in the data. 
1. We use one-hot encoding on the neighbourhood_group, neighbourhood, room_type variables resulting in 235 features in total.
2. Next, we use standard scale to normalise the features.

  <h3>Model selection: </h3>
  We experiment with various machine learning models which includes 
1. Classic ML mosels such as ridge regression, lasso regression, decision tree regression.
2. Ensemble Bagging Methods such as random forest, extra trees.
3. Ensemble Boosting methods such as gradient boosting, XGBoost, and CATBoost regression. 
We use the scikit-learn library to implement the models.

  <h3>Model evaluation: </h3>
  We evaluate the performance of each model using various metrics, such as r2 score for test data, mean squared error (MSE), mean absolute error (MAE), and R-mean squared error. We also use cross-validation to assess the generalization performance of the models.
  
  
  
 
   <img src="https://user-images.githubusercontent.com/127759119/226189914-839ab9f3-4548-422c-bb92-185fb6aa4d31.png" width="500">
  
  
  
  <h3>Hyperparameter tuning:</h3>
  We use random search on various parameters (such as learning rate, max_depth, n_estimators, min_samples_split, min_samples_leaf, subsample, max_features, and many more) to find the optimal hyperparameters for each model, in order to improve the model's performance.


  <h3>Interpretation: </h3>
  We use the yellowbrick library to visualize the results and comparison among different models using each metric of evaluation, which helps us to understand the best model according to a particular metric for the prediction of Airbnb prices.
  
  <img src="https://user-images.githubusercontent.com/127759119/226191125-3eabaf3a-cd78-48a9-a2a1-41a2e67e4e90.png" width="1000">

## Results
From all the utilised models, we clearly see that the ensemble methods outperform the classic ML models. Moreover, XGBoost and CATBoost appeared as the best models for prediction of prices. The hypertuning of the XGBoost model resulted in an accuracy of 62.87% on the test dataset and 78.4% on the train data, and the best RMSE value of 0.42, among all models.

## Prerequisites
<b>To run this project, you need to have Python 3.x installed on your system. You also need to install the following Python libraries:</b>

1. Pandas
2. Numpy
3. Matplotlib
4. Seaborn
5. Scikit-Learn 
6. XGBoost
7. CATBoost
8. Yellowbrick

  <b>You can install these libraries using the following command:</b>
'pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost yellowbrick'

## Usage
Clone the repository using git clone https://github.com/dasjyotishka/Predicting-Airbnb-Prices-in-New-York-City.git
Navigate to the directory using cd Predicting-Airbnb-Prices-in-New-York-City
Open the Notebook using jupyter notebook/google colab
Run the cells in the notebook to read data, analyse it along with training and evaluation of different models.

## Acknowledgements
This project was inspired by the Kaggle competition "New York City Airbnb Open Data" <a>(https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)</a>.

## License
The code in this repository is licensed under the MIT license. See LICENSE for more details.
