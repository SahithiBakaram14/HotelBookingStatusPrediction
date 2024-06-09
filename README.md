# HOTEL BOOKING PREDICTION
- In this project, firstly the data extraction, data preprocessing and Exploratory Data Analysis is done followed by machine learning modeling,
- and finally, the machine learning model with the best accuracy for the taken data is locally deployed using flask.

## Problem Statement
As travel demand increases, getting a quality hotel room becomes harder. Using “Hotel Booking Demand database”, we use machine learning to 
create an approach for hotel booking by utilizing data and booking attitudes and behaviors. Travelers are increasingly impacted by the daily 
trend of hotel cancellations during border times. Using “Hotels booking Demand” analysis we can know the Hotel Cancellation Predictions, thereby
users may choose whether to book a hotel
in advance or not. Further, it will also address some of the following questions:
- Which hotel is the most popular?
- What elements affect hotel cancellations?

## Phases of the Project
### Phase1

The aim of this phase is to collect the dataset and do data cleaning followed by exploratory data analysis to find out the characteristics of each
response variable and the relationship patterns between the response variables and predictor variable.

### Phase2

Here the chosen dataset is the Classification problem with the prediction inference, so using any five classification machine learning models training,
testing and prediction are done.

### Phase3

Building the web application to the most accurate machine learning model for the dataset is done using Flask to predict the hotel booking cancellation 
given the user input.

## Overall Procedure

- Collection of raw data

  The “Hotels Booking Demand” data is taken from Kaggle.com using the reference https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand .
  The data was first published in the article Hotel Booking Demand Datasets, written by Nuno Antonio, Ana Almeida, and Luis Nunes for Data in Brief,
  Volume 22, February 2019.


- Data Processing
  
  The collected raw data is extracted, analyzed and processed to know the features information and description of the dataset by performing basic operations.

- Data Cleaning
    - The process of data cleaning is started with the deletion of duplicates.
    - Then the identification and imputation of missing values are done.
    - Deleting unwanted rows and columns.
    - Converting all columns to proper data types and precision
    - Identifying the Outliers
    - Removing the outliers 
    - Univariate or Multivariate Non-Graphical Techniques
      - Shape of the Distribution
      - Measures of Center: Mean, Median.
      - Modality:Modality gives the most frequent value in the dataset. We have calculated the mode 
values for each feature.
      - Quantiles:We split the data into four parts using quantiles i.e., 25%, 50%, 75%, max. 
25% refers to the first quartile, 50% refers to the median, 75% refers to the third quartile.

- Exploratory Data Analysis
    - Univariate or Multivariate Graphical Techniques
    - Visualization of dataset is done with the pie plots to know the distribution of each unique value for some of the features
    - Count plots and bar plots for the multivariate analysis of the features.
    - Histplots to know the shape of distribution of each dataset feature.

- Lets seperate the response variables to X and predictor variable to Y which is is_cancelled
- Now the encoding is done for categorical features  and normalisation is done for numerical features.

- Splitting of the dataset
  
The dataset is split into training and testing with the 80% and 20% of the data respectively.
- Model building
  
Creating the model using the following five classification algorithms
    
    - Naive Bayes
    - Logistic Regression
    - KNN
    - Decision tree
    - Random forest
  
Models are evaluated using the confusion matrix and acurracy.The graph with comparision of the accuray of each model is plotted.

- The Random forest model has the highest accuracy so feature importance for this model is drawn.Inputs are considered from the drawn feature importance.

The web application is built with the Flask to predict the hotel booking cancellation using the random forest model 

- Phase3 folder constitutes the files related to web application built using flask.
    - data file
    - templates folder with input.html and result.html
    - app.py with code related to the built the model which is saved as pickle file that is used for the prediction of user input
    - model.pkl the generated pickle file

## Execution steps:

In app.py we have mentioned the local path of files
- For the dataset in 6th line 
- For pickle file in 37 and 43 lines
- After running the python file app.py, result in the terminal shows that the web application is
Running on http://127.0.0.1:5000

Let us go to that link and give the user input, then click on submit to see the prediction of model.
