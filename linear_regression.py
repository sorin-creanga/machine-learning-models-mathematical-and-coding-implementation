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
experience_year = experience_year[:len(salaries)]


mean_salaries = sum(salaries)/len(salaries)
mean_experience_year = sum(experience_year)/len(experience_year)

"""
print(
    f"Mean values are:\n"
    f"Mean Salaries: {round(mean_salaries,2)}\n"
    f"Mean Years of Experience: {round(mean_experience_year,2)}"
)
"""

st_salaries = math.sqrt((sum((_ - mean_salaries)**2 for _ in salaries))/len(salaries))
st_experience_years = math.sqrt((sum((_ - mean_experience_year)**2 for _ in experience_year))/len(experience_year))


covariance = (
    sum(
        (a - mean_salaries) * (b - mean_experience_year)  
        for a, b in zip(salaries, experience_year)
    )  
    / (len(salaries) - 1)
)

pearson_correlation= covariance / (st_salaries*st_experience_years)

print(f"The Pearson Coef: {round(pearson_correlation,2)}")

"""
Please find below a basic visual representation of the the Salaries and Experience Years
variables in a scatter plot.

The red line describes the slope of the relationship showcasing the rate at which the salaries
grow as experience grows.

Feel free to take the chart out of the comment section and try it out on your machine.

**Chart**

"""