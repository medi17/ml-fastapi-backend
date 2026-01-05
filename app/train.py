import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("creditcard.csv")

X = df.drop(columns=['Class']) 
y = df['Class']


logistic_model = LogisticRegression(max_iter=1000)
tree_model = DecisionTreeClassifier()


logistic_model.fit(X, y)
tree_model.fit(X, y)

joblib.dump(logistic_model, "models/logistic_model.joblib")
joblib.dump(tree_model, "models/decision_tree_model.joblib")

print("Models trained and saved successfully!")
