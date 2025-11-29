"""
    In this python file the aim is to replicate mathematically and manually the work complexity of the LinearRegression() model
    from the SKLEARN.model library.

    The aim is to develop a deep understating of how the model works under the hood, before rellying on pre-build packages.

"""

import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv(r"C:\Users\sorin.creanga\Desktop\Math_for_ML_Models\Salary_Data[1].csv", index_col=0)


salaries = [_ for _ in data["Salary"] if not math.isnan(_)]
experience_year = [_ for _ in data["Years of Experience"]if not (math.isnan(_))]


