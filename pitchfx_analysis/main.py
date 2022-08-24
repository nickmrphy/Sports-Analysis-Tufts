from pybaseball import statcast
import numpy as np
import pandas as pd
import sklearn
import load
import process
import math
import operator

from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Perceptron, Ridge
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import PolynomialFeatures

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
# %matplotlib inline
plt.style.use('seaborn') # pretty matplotlib plots
import sys
import model


def main():
    print("Hello!")
    des = input("Enter 'A' if you'd like to run the preset analysis or 'B' for custom")
    if des == 'A':
        x = '2017-06-25'
        y = '2017-07-25'
        data = statcast(start_dt=x, end_dt=y)
        new_data, x, y = load.load(data)
    elif des == 'B':
        x = input("Please enter the start date to collect data from in 'YYYY-MM-DD' format")
        y = input("Please enter the end date to collect data to in 'YYYY-MM-DD' format")
        data = statcast(start_dt=x, end_dt=y)
        new_data, x, y = load.load(data)
    return 0


if __name__ == "__main__":
    main()
