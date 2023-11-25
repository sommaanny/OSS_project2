import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

def sort_dataset(dataset_df):
	result = dataset_df.sort_values(by='year')
	return result

def split_dataset(dataset_df):	
	X = dataset_df.drop('salary', axis=1)
	Y = dataset_df['salary']
	Y = Y * 0.001
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10193413)
	return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
	result = dataset_df.loc[:, ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]
	return result

def train_predict_decision_tree(X_train, Y_train, X_test):
	rf_reg = DecisionTreeRegressor()
	rf_reg.fit(X_train, Y_train)
	predict = rf_reg.predict(X_test)
	return predict

def train_predict_random_forest(X_train, Y_train, X_test):
	rf_reg = RandomForestRegressor()
	rf_reg.fit(X_train, Y_train)
	predict = rf_reg.predict(X_test)
	return predict

def train_predict_svm(X_train, Y_train, X_test):
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	
	rf_reg = SVR()
	rf_reg.fit(X_train_scaled, Y_train)
	predict = rf_reg.predict(X_test_scaled)
	return predict

def calculate_RMSE(labels, predictions):
	return np.sqrt(np.mean((predictions - labels)**2))

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))