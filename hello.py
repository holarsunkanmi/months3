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

pd_data = "../Months 3/data/Telco-Customer-Churn.csv"
read_data = pd.read_csv(pd_data)
read_data.describe()
read_data.columns
read_data.head()
read_data.info()
read_data["TotalCharges"] = pd.to_numeric(read_data["TotalCharges"], errors="coerce")
read_data["TotalCharges"].fillna(read_data["TotalCharges"].median(), inplace=True)
read_data = read_data.drop(['customerID'], axis=1)
y = read_data.Churn
X = read_data.drop(['Churn'], axis=1)
X = pd.get_dummies(X, drop_first=True)

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
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": True,
    "criterion": "gini",
    "class_weight": None,
}

# LOGISTIC REGRESSION

with mlflow.start_run():
    ClassTree = DecisionTreeClassifier(**params)
    ClassTree.fit(Train_X, Train_y)
    ClassTree.predict(Test_X)  
    preds = ClassTree.predict(Test_X) 
    accuracy_score(Test_y, ClassTree.predict(Test_X))
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
   

    infer_signatures = infer_signature(Train_X, ClassTree.predict(Test_X))
    model_info = mlflow.sklearn.log_model(
        sk_model=ClassTree,
        name="Churns Tree",
        signature=infer_signature,
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
    accuracy_score(Test_y, RanForest.predict(Test_X))
    accuracy = accuracy_score(Test_y, RanForest.predict(Test_X))
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_artifact(pd_data)
    mlflow.log_params(params)


    infer_signature = infer_signature(Train_X, RanForest.predict(Test_X))
    model_info = mlflow.sklearn.log_model(
        sk_model=RanForest,
        name="Churns Forest",
        signature=infer_signature,
        registered_model_name="Churns Forest",
    )

    load_model = mlflow.sklearn.load_model(model_info.model_uri)
    load_model.predict(Test_X)
    accuracy_score(Test_y, load_model.predict(Test_X))















