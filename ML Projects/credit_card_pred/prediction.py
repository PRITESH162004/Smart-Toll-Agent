import pandas as pd

from sklearn.linear_model import LogisticRegression

data = {
    'income': [20000,30000,50000,80000,15000,25000,40000,10000],
    'loan_amount': [5000,10000,15000,20000,25000,30000,45000, 10000],
    'default': [0,0,0,0,1,1,1,0]
}

df = pd.DataFrame(data)

x = df[['income', 'loan_amount']]
y = df['default']

model = LogisticRegression()
model.fit(x, y)

new_applicant = pd.DataFrame([[30000,45000]], columns=['income', 'loan_amount'])

prediction = model.predict(new_applicant)

probability = model.predict_proba(new_applicant)[0][1]

print(f"Prediction: {'Risky' if prediction[0] == 1 else 'SAFE'}")
print(f"Risk Probability: {probability*100:.2f}%")