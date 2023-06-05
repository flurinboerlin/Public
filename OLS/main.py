from Functions.OLS import my_regression
# from 01_Functions.OLS.py import my_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

repoDir = "C:/Users/fluri/OneDrive/Documents/github/Public/"

cars = pd.read_csv(f'{repoDir}OLS/02_Data/cars.csv')


regression_equation = "speed = dist + random"

regression_object = my_regression(data = cars, regression_equation = regression_equation)

regression_object.regression_equation
regression_object.summary_statistics

print('done')