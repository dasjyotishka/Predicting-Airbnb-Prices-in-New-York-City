# Predicting Airbnb Prices in New York City
This project aims to predict the prices of Airbnb listings in New York City based on various features using regression models. It uses a dataset of Airbnb listings in New York City, which includes information such as location, room type, number of reviews, availability, and host name. 

## Dataset
The dataset used in this project is available on <b><u><a>https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data</a></u></b>. It contains information on over 48,000 Airbnb listings in New York City. The dataset includes the following features:
1. id: the unique ID of the listing
2. name: the name of the listing
3. host_id: the unique ID of the host
4. host_name: the name of the host
5. neighbourhood_group: the borough in which the listing is located (e.g., Brooklyn, Manhattan, Queens, Staten Island, Bronx)
6. neighbourhood: the neighbourhood in which the listing is located
7. latitude: the latitude of the listing
8. longitude: the longitude of the listing
9. room_type: the type of room (e.g., private room, entire home/apt, shared room)
10. price: the price per night of the listing
11. minimum_nights: the minimum number of nights required to book the listing
12. number_of_reviews: the number of reviews the listing has received
13. last_review: the date of the last review
14. reviews_per_month: the number of reviews per month
15. calculated_host_listings_count: count of listings per host_id
16. availability_365: the number of days the listing is available for booking in the next 365 days

## Approach
The approach used in this project involves data preprocessing, exploratory data analysis, feature engineering, and model training and evaluation. The main machine learning models used in this project are linear regression, decision tree, and random forest.
The approach used in this project involves several steps ranging from data preprocessing, exploratory data analysis, feature engineering, to model selection, training and evaluation. The following are the main steps involved:

###Exploratory data analysis (EDA): We use visualization techniques to better understand the distribution and relationship of the dataset features, and to gain insights about the data. The EDA helps us to decide which features to use in the modeling process. The process is as follows:
1. We visualize different features (such as room_type, neighbourhood_group, and neighbourhood), and there relationship with the target variable (price). ![image](https://user-images.githubusercontent.com/127759119/226188644-cbea874b-c25b-4e29-995b-93873412e593.png)![image](https://user-images.githubusercontent.com/127759119/226188696-3c913db1-8cc7-444f-8479-46b0d33f95d4.png)
2. We also analyse the relation between the variable themselves (such as Distribution of neighbourhood_group and room_type with Respect to Latitude and Longitude)![image](https://user-images.githubusercontent.com/127759119/226188719-b39a4476-d96e-4d7a-bc35-f8c220191a0a.png)![image](https://user-images.githubusercontent.com/127759119/226188732-592ab329-56c0-4253-af06-d453d0f03f97.png)
3. Next, we visualize the distribution of the target variable (price) and apply log transformations to create a normal distribution.![image](https://user-images.githubusercontent.com/127759119/226188754-893484a7-2c7d-45bc-b782-37eab9ab771d.png)
4. Next, we detect the outliers in target variable (price) and remove these.![image](https://user-images.githubusercontent.com/127759119/226188780-5c8db7be-c786-45b7-82d9-005140a6492e.png)
5. Finally, we visualise the correlation among different numerical features of the dataset using seaborn heatmap.![image](https://user-images.githubusercontent.com/127759119/226188807-abf3acda-a3d0-451e-9d82-ee2c3132812d.png)

###Data cleaning: The dataset contains missing values and some outliers, which need to be addressed. We remove the records with missing values and use various techniques, such as Z-score analysis, to identify and remove outliers.

###Feature engineering: We create new features from the existing ones to help the machine learning algorithms better capture the patterns in the data. For example, we create new features such as "price per bedroom" and "price per bathroom".

###Model selection: We experiment with various machine learning models, including linear regression, decision tree regression, and ensemble methods such as random forest, extra trees, and gradient boosting regression. We use the scikit-learn library to implement the models.

###Model evaluation: We evaluate the performance of each model using various metrics, such as mean squared error (MSE), mean absolute error (MAE), and R-squared. We also use cross-validation to assess the generalization performance of the models.

###Hyperparameter tuning: We use grid search and random search to find the optimal hyperparameters for each model, in order to improve the model's performance.

###Interpretation: We use the yellowbrick library to visualize the feature importances of the models, which helps us to understand the most important features that contribute to the prediction of Airbnb prices.

## Prerequisites
To run this project, you need to have Python 3.x installed on your system. You also need to install the following Python libraries:

1. Pandas
2. Numpy
3. Matplotlib
4. Seaborn
5. Scikit-Learn (pre-processing, model_selection, metrics)
6. Sklearn linear_model, DecisionTreeRegressor
7. Sklearn ensemble (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor)
8. XGBoost
9. CATBoost
10. Yellowbrick

You can install these libraries using the following command:
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
