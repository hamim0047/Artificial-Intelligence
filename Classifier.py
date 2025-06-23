import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_iris, y_iris,
    train_size=0.75,
    random_state=1,
    shuffle=True
)

scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(Xtrain_scaled, ytrain)
lr_pred = lr_model.predict(Xtest_scaled)
print("Logistic Regression Accuracy:", accuracy_score(ytest, lr_pred))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(Xtrain, ytrain)
dt_pred = dt_model.predict(Xtest)
print("Decision Tree Accuracy:", accuracy_score(ytest, dt_pred))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(Xtrain, ytrain)
rf_pred = rf_model.predict(Xtest)
print("Random Forest Accuracy:", accuracy_score(ytest, rf_pred))

# Support Vector Machine (SVM)
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(Xtrain_scaled, ytrain)
svm_pred = svm_model.predict(Xtest_scaled)
print("Support Vector Machine (SVM) Accuracy:", accuracy_score(ytest, svm_pred))

# MLP Classifier
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(max_iter=1000)
mlp_model.fit(Xtrain_scaled, ytrain)
mlp_pred = mlp_model.predict(Xtest_scaled)
print("MLP Classifier Accuracy:", accuracy_score(ytest, mlp_pred))

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(Xtrain_scaled, ytrain)
knn_pred = knn_model.predict(Xtest_scaled)
print("K-Nearest Neighbors (KNN) Accuracy:", accuracy_score(ytest, knn_pred))
