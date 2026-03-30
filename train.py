import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# load data
data = pd.read_csv("data.csv")

# features and target
X = data[["hours", "attendance", "marks"]]
y = data["result"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = LogisticRegression()
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")