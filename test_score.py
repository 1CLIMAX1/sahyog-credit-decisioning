import pandas as pd
from src.score import score_application

df = pd.read_csv("data/loan_applications.csv")

app = df.sample(1).iloc[0].to_dict()

print("\nINPUT APPLICATION:")

for key, value in app.items():
    print(f"{key}: {value}")


print("\nMODEL OUTPUT:")

print(score_application(app))