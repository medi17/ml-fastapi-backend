import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. Load your friend's data
df = pd.read_csv("creditcard.csv")

# 2. Separate features (X) and the result (y)
# Assuming the last column is what you want to predict (e.g., 'Class')
X = df.drop(columns=['Class']) 
y = df['Class']

# 3. Create the models
logistic_model = LogisticRegression(max_iter=1000)
tree_model = DecisionTreeClassifier()

# 4. Train them
logistic_model.fit(X, y)
tree_model.fit(X, y)

# 5. Save them into your 'models' folder
joblib.dump(logistic_model, "models/logistic_model.joblib")
joblib.dump(tree_model, "models/decision_tree_model.joblib")

print("Models trained and saved successfully!")