"""
    In this python file the aim is to replicate mathematically and manually the work complexity of the LinearRegression() model
    from the SKLEARN.model library.

    The aim is to develop a deep understading of how the model works under the hood, before rellying on pre-build packages.

"""

import pandas as pd
import matplotlib.pyplot as plt
import math


data = pd.read_csv(r"C:\Users\sorin.creanga\Desktop\Math_for_ML_Models\Salary_Data[1].csv", index_col=0)


cleaned_data = data.dropna(subset=['Salary', 'Years of Experience'])


salaries = cleaned_data["Salary"].tolist()
experience_year = cleaned_data["Years of Experience"].tolist()

print(f"Final Aligned Data Length: {len(salaries)}")


mean_salaries = sum(salaries)/len(salaries)
mean_experience_year = sum(experience_year)/len(experience_year)

"""
print(
    f"Mean values are:\n"
    f"Mean Salaries: {round(mean_salaries,2)}\n"
    f"Mean Years of Experience: {round(mean_experience_year,2)}"
)
"""


st_salaries = math.sqrt((sum((_ - mean_salaries)**2 for _ in salaries))/(len(salaries) - 1))
st_experience_years = math.sqrt((sum((_ - mean_experience_year)**2 for _ in experience_year))/(len(experience_year) - 1))


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

slope_m = covariance / (st_experience_years**2)
b_for_slope = mean_salaries-(slope_m*mean_experience_year)
Y_predicted_manual = [(slope_m * x_val) + b_for_slope for x_val in experience_year]

#Regression Line


slope_m = covariance / (st_experience_years**2)


# Formula: b = Mean(Y) - m * Mean(X)
b_for_slope = mean_salaries - slope_m * mean_experience_year


# Formula: Y_pred = m * X + b
Y_predicted_manual = [slope_m * x + b_for_slope for x in experience_year]

print(f"Calculated Slope (m): {slope_m}")
print(f"Calculated Intercept (b): {b_for_slope}")



plt.figure(figsize=(10,6))


plt.scatter(experience_year, salaries, color = "blue", label='Data Points')

plt.scatter(mean_experience_year, mean_salaries, color='green', marker='D', s=100, label='Mean Point')


plt.plot(experience_year, Y_predicted_manual, color="red", label="Regression line")

plt.title('Linear Regression: Salary vs. Years of Experience', size=20, color = "blue")
plt.xlabel('Years of Experience', size = 16)
plt.ylabel('Salary', size = 16)
plt.legend()
plt.grid(True)
plt.show()