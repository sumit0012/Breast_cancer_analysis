import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import skops.io as sio
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer



# Load the breast cancer dataset
df = pd.read_csv(r'data_cancer.csv')

X=df.drop(["diagnosis"],axis=1)
Y=df["diagnosis"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

num_col = [0,1,2,3,4,5,6,7,8,9]

transform = ColumnTransformer(
    [
        
        ("num_scaler", StandardScaler(), num_col)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model",LogisticRegression()),
    ]
)

# Standardize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.values)
# X_test = scaler.transform(X_test.values)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train.values, y_train)

# Make predictions
y_pred = model.predict(X_test.values)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average="macro")

print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.savefig("./Results/model_results.png", dpi=120)

# Write metrics to file
with open("./Results/metrics.txt", "w", encoding="utf-8") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}")

# Saving the model file
sio.dump(model, "./Model/cancer_pipeline.skops")
