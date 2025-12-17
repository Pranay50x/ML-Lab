# Program 6

```python
import numpy as np 
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(42)
x, y = make_regression(n_samples=300, n_features=1, noise=10, random_state=42)

y = y + 2*x.flatten()**2 - 3*x.flatten()**3 + np.random.normal(0,15, len(y))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

print("Dataset overview: ")
print("Test Shape: ", x_test.shape)
print("train Shpae: ", x_train.shape)

degrees = [1,2,3, 6]
fig, axes = plt.subplots(2,2, figsize=(15,10))

for degree, ax in zip(degrees, axes.flatten()): 
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    rmse, r2 = root_mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)

    x_curve = np.linspace(x.min(), x.max(), num=300).reshape(-1,1)
    ax.scatter(x_test, y_test, label='Actual')
    ax.plot(x_curve, model.predict(x_curve), label='Predicted', color='red')
    ax.set_title(f'Degree: {degree}\nR2: {r2:.2f}, RMSE: {rmse:.2f}')
    ax.legend()

plt.tight_layout()
plt.show()
```

# Program 7

```python
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve)

df = pd.read_csv('titanic.csv')
df.rename(columns={'Siblings/Spouses Aboard': 'SibSp', 'Parents/Children Aboard': 'Parch'}, inplace= True)
x = df[['Pclass', 'Sex' ,'Age', 'SibSp', 'Parch', 'Fare']].copy()
y = df['Survived']

x = pd.get_dummies(x, columns=['Sex'], drop_first=True)

print("Dataset overview: ")
print(f"Dataset shape: {df.shape}")
print("Feature Columns", list(x.columns))
print(f"Survival Rate: : {y.mean():.2f}")
print("First 5 rows: ")
print(df.head())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

print("Performance metrics: ")
print(f"Accuracy {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report: ")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, cmap='Reds', fmt='d', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title("Confusion Matrix")
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

fpr, tpr, _ = roc_curve(y_test,y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr,tpr, color='red', lw=2, label=f'AUC = {roc_auc_score(y_test,y_prob):.2f}')
plt.plot([0,1],[0,1], 'k--', label='Random')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

# Program 8 

```python
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix)

iris = load_iris(as_frame=True)
x = iris.data
y = iris.target

print("Dataset shape: ",x.shape)
print("Target Feats: ", iris.target_names)
print("Distribution: ")
print(y.value_counts())

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42, stratify=y)

model = make_pipeline(StandardScaler(), KNeighborsClassifier())
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

print("Wrong pred: ")
if(len(wrong)>0) : 
    print(wrong)
else:
    print("no wrong pred")

print("\nCorrect Preds: ")
print(correct.head())

plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test, y_pred),annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.show()
```

# Program 9 

```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_breast_cancer(as_frame=True)
x, y = data.data, data.target

print("Dataset overview: ")
print("\nSHape: ", x.shape)
print("Target classes: ", list(data.target_names))
print("Class dist: ")
print(y.value_counts())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

new_sample = x_test.iloc[[0]]
pred_name = data.target_names[model.predict(new_sample)[0]]

print("Metrics: ")
print(f"Accuracy score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report: ")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("new Sample result: ")
print(f"Features: {new_sample.values[0][:4]} .. truncated")
print(f"\nPredicted class: {pred_name}")

plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix(y_test, y_pred), cmap='Reds', annot=True, fmt='d', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion matrix')
plt.tight_layout()
plt.show()
``` 

# Program 10 

```python
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 
import numpy as np

np.random.seed(42)
x = np.vstack([
    np.random.multivariate_normal([0,0], np.eye(2), 50), 
    np.random.multivariate_normal([5,5], np.eye(2), 50)
])
N, D = x.shape
K = 2

mean = np.random.rand(K,D) *5
covs = np.array([np.eye(D)]*K)
weights = np.ones(K)/K 
resps = np.zeros((N,K))

for i in range(10):

    for k in range(K):
        #e step
        resps[:,k] = weights[k] * multivariate_normal.pdf(x, mean[k], covs[k])
    resps /= resps.sum(axis =1, keepdims=True)

    for k in range(K): 
        nk = resps[:,k].sum()
        mean[k] = (resps[:,k,None] *x).sum(axis=0)/nk
        diff = x - mean[k]
        covs[k] = np.dot((resps[:,k,None] * diff ).T, diff)/nk
        weights[k] = nk/N
    plt.figure(figsize=(10,5))
    plt.scatter(x[:,0], x[:,1], c=np.argmax(resps, axis=1))
    plt.scatter(mean[:,0], mean[:,1], c='red', marker='x')
    plt.title(f"Iteration: {i+1}")
    plt.tight_layout()
    plt.show()

final = np.argmax(resps, axis=1)
print("Final: ")
print(final)
```

# Program 11

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,RandomForestClassifier,VotingClassifier)

data = load_breast_cancer(as_frame=True)
x,y = data.data, data.target

print("Dataset overview: ")
print("Shape: ",x.shape)
print("Feature cols: ", data.target_names)
print("Class dist: ")
print(y.value_counts())

models = {
    'Voting (DT, LR, SVC)': VotingClassifier(estimators=[
        ('dt', DecisionTreeClassifier(max_depth=3)), ('lr',LogisticRegression(max_iter=3000)), ('sv',SVC())
    ]), 
    'BaggingClassifier' : BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
    'AdaBoost' : AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME'),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42), 
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42)
}

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42, stratify=y)

print("Model perf metrtics: ")

for name,model in models.items():
    model.fit(x_train,y_train)

    acc = accuracy_score( y_test,model.predict(x_test))

    cv = cross_val_score(model, x_train, y_train, cv=5)

    print('\nModel: ', name)
    print(f"Accuracy Score: {acc:.4f}")
    print(f"Cross Val Score: {cv.mean():.4f} (+/- {cv.std():.4f})")
```