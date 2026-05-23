import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

data_path = os.path.join('data','loan_data.csv')
df = pd.read_csv(data_path)

x = df[['income','loan_amount']]
y = df['default']

model = LogisticRegression()
model.fit(x,y)

model_path = os.path.join('models','credit_model.pkl')
joblib.dump(model, model_path)

print(f"Step 1 Success: Data loaded from {data_path} and model saved at {model_path}")


