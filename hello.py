import mlflow

print (mlflow.tracking.get_tracking_uri())
mlflow.set_tracking_uri("file:./mlflow_runs")
mlflow.set_experiment("Months3_Experiment")
mlflow.sklearn.autolog()



import pandas as pd;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature
import joblib

pd_data = "data/telco-Customer-Churn.csv"
read_data = pd.read_csv(pd_data)
read_data.describe()
read_data.columns
read_data.head()
read_data.info()
read_data["TotalCharges"] = pd.to_numeric(read_data["TotalCharges"], errors="coerce")
read_data["TotalCharges"] = read_data["TotalCharges"].fillna(
    read_data["TotalCharges"].median()
)
read_data = read_data.drop(['customerID'], axis=1)
y = read_data.Churn
X = read_data.drop(['Churn'], axis=1)
X = pd.get_dummies(X, drop_first=True)
joblib.dump(X.columns.tolist(), "models/features.pkl")

ClassTree = DecisionTreeClassifier(random_state=42, max_leaf_nodes=50)
ClassTree.fit(X, y)
ClassTree.predict(X)
accuracy_score(y, ClassTree.predict(X))

# TRAIN DATA AND SPLIT

Train_X, Test_X, Train_y, Test_y = train_test_split(X, y, test_size=0.25, random_state=42)

#PARAMS

params = {
    "random_state": 42,
    "max_leaf_nodes": 50,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "criterion": "gini",
    "class_weight": None,
}

# LOGISTIC REGRESSION

best_accuracy = 0
best_model_name = ""
best_model = None

with mlflow.start_run():
    ClassTree = DecisionTreeClassifier(**params)
    ClassTree.fit(Train_X, Train_y)
    ClassTree.predict(Test_X)  
    preds = ClassTree.predict(Test_X) 
    accuracy = accuracy_score(Test_y, preds)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = "Decision Tree"
        best_model = ClassTree

    precision = precision_score(Test_y, preds, pos_label="Yes")
    recall = recall_score(Test_y, preds, pos_label="Yes")
    f1 = f1_score(Test_y, preds, pos_label="Yes")
    accuracy = accuracy_score(Test_y, ClassTree.predict(Test_X))
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_artifact(pd_data)
    mlflow.log_params(params)
   

    signatures = infer_signature(Train_X, ClassTree.predict(Test_X))
    model_info = mlflow.sklearn.log_model(
        sk_model=ClassTree,
        name="Churns Tree",
        signature=signatures,
        registered_model_name="Churns Tree",
    )

    loaded_model = mlflow.sklearn.load_model(model_info.model_uri)
    loaded_model.predict(Test_X)
    accuracy_score(Test_y, loaded_model.predict(Test_X))




with mlflow.start_run():

    RanForest = RandomForestClassifier(**params)
    RanForest.fit(Train_X, Train_y)
    RanForest.predict(Test_X)
    preds = RanForest.predict(Test_X)
    precision = precision_score(Test_y, preds, pos_label="Yes")
    recall = recall_score(Test_y, preds, pos_label="Yes")
    f1 = f1_score(Test_y, preds, pos_label="Yes")
    accuracy = accuracy_score(Test_y, preds)

# Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = "Random Forest"
        best_model = RanForest
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_artifact(pd_data)
    mlflow.log_params(params)


    signature = infer_signature(Train_X, RanForest.predict(Test_X))
    model_info = mlflow.sklearn.log_model(
        sk_model=RanForest,
        name="Churns Forest",
        signature=signature,
        registered_model_name="Churns Forest",
    )

    load_model = mlflow.sklearn.load_model(model_info.model_uri)
    load_model.predict(Test_X)
    accuracy_score(Test_y, load_model.predict(Test_X))

    mlflow.set_tag("best_model", best_model_name)
    mlflow.log_metric("best_accuracy", best_accuracy)


joblib.dump(best_model, "models/model1.pkl")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy}")












