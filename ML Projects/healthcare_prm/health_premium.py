import numpy as np
# import pandas as pd

age = np.array([18,25, 35,45,55])
premium = np.array([1500,18000, 21000, 24000,29000])

def train_basic_regression(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x-x_mean)*(y-y_mean))
    denominator = np.sum((x-x_mean)**2)

    m= numerator/denominator

    b = y_mean - (m * x_mean)

    return m,b

slope, intercept = train_basic_regression(age,premium)

def predict_premium(user_age):
    return (slope* user_age)+ intercept


test_age = 18
result = predict_premium(test_age)

print(f"---Healthcare Predictor (Basic)---")
print(f"Cost increase per year (Slope): ₹{slope:.2f}")
print(f"Base insurance cost (Intercept): ₹{intercept:.2f}")
print(f"Predicted Premium for age {test_age}: ₹{slope:.2f} ")
