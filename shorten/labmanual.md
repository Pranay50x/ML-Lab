## Prog1 EDA on Housing Dataset

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# 1. Load Data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

print(f"Dataset shape: {df.shape}")
print(f"\\nColumns: {list(df.columns)}")
print("\\nStructure: "); print(df.describe())

# 2. Histograms with Mean/Median lines
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('California Housing Distribution', fontsize=16, fontweight='bold')

for ax, col in zip(axes.flatten(), df.columns):
    df[col].hist(bins=30, ax=ax, alpha=0.7, edgecolor='black')
    ax.axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
    ax.axvline(df[col].median(), color='green', linestyle='--', label='Median')
    ax.set_title(f'{col} Distribution', fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.show()

# 3. Boxplots for Outliers
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Outlier Detection', fontsize=16, fontweight='bold')

for ax, col in zip(axes.flatten(), df.columns):
    df.boxplot(column=col, ax=ax, color='red')
    
    # Calculate outliers for title info
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    
    ax.set_title(f"{col} - {len(outliers)} outliers ({len(outliers)/len(df):.1%})", fontweight='bold')

plt.tight_layout()
plt.show()
```


## Prog2 Correlation Analysis

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

# 1. Load Data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
print('Iris dataset shape:', df.shape)
print("\nFirst rows: "); print(df.head())

# 2. Correlation Matrix
numeric_df = df.drop('species', axis=1)
corr_m = numeric_df.corr()

print("\nCorr Matrix: "); print(corr_m.round(2))

# 3. Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_m, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Iris Dataset Heatmap', fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.show()

# 4. Strong Correlations Analysis
print("Corr analysis: ")
print("Strong correlation (|r| > 0.7): ")

# Extract pairs with high correlation (excluding diagonal)
strong_pairs = corr_m.unstack().sort_values(ascending=False)
strong_pairs = strong_pairs[(abs(strong_pairs) > 0.7) & (abs(strong_pairs) < 1.0)]

# Filter duplicates and print
seen = set()
for (f1, f2), val in strong_pairs.items():
    if (f2, f1) not in seen:
        print(f"{f1} - {f2}: {val:.3f}")
        seen.add((f1, f2))
```

## Prog3 PCA

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
x_scaled = StandardScaler().fit_transform(iris.data)
print("Shape: ", iris.data.shape)
print("Features: ",iris.feature_names)
print("Target classes: ", iris.target_names)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

print("\nPCA Results: ")
print("Explained variance ratio: ", pca.explained_variance_ratio_)
print("Total variance explained: {:.2f}%".format(sum(pca.explained_variance_ratio_) * 100))

plt.figure(figsize=(10, 8))
colors = ['navy', 'red', 'darkorange']

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(x_pca[iris.target == i, 0], x_pca[iris.target == i, 1], 
                color=color, alpha=0.8, lw=2, label=target_name)

plt.legend(loc='best', shadow=False)
plt.title('PCA of Iris Dataset')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(alpha=0.3)
pc1_top = iris.feature_names[np.argmax(abs(pca.components_[0]))]
pc2_top = iris.feature_names[np.argmax(abs(pca.components_[1]))]
plt.figtext(0.02, 0.02, f"PC1 influenced by: {pc1_top}\nPC2 influenced by: {pc2_top}", 
            bbox=dict(facecolor='white', alpha=0.7))

plt.show()

print("PCA Details: ")
print("PCA1 Loadings: ")
for i,col in enumerate(df.columns): 
    print(f"{col}: {pca.components_[0][i]:.3f}")

print("PCA2 Loadings: ")
for i,col in enumerate(df.columns): 
    print(f"{col} : {pca.components_[1][i]:.3f}")

```

## Prog4 Candidate Elimination

```python
import numpy as np 
import pandas as pd

data = pd.DataFrame(pd.read_csv('./ENJOYSPORT.csv'))

concept = np.array(data.iloc[:, 0: -1])
target = np.array(data.iloc[:, -1])

specific_h = concept[0].copy()
print("Specific h: ",specific_h)

general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
print("General: ", general_h)

for i, h in enumerate(concept): 
    
    if target[i]=="yes":
        for x in range(len(specific_h)): 
            if h[x]!=specific_h[x]: 
                specific_h[x] ='?'
                general_h[x][x] = '?'
    else: 
        for x in range(len(specific_h)): 
            if h[x] != specific_h[x]: 
                general_h[x][x] = specific_h[x]
            else:  
                general_h[x][x] = "?"
    print("Step",i+1)
    print("Specific_h: ", specific_h)
    print("General_h: ", general_h)
    
indices = [i for i, val in enumerate(general_h) if val== ['?' for _ in range(len(specific_h))]]
for i in sorted(indices, reverse=True): 
    del general_h[i]

print("\nFinal ", specific_h)
print("\nFinal: ", general_h)
```

## Prog Linear Regression: 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing

# 1. Load Data
housing = fetch_california_housing()
X, y = pd.DataFrame(housing.data, columns=housing.feature_names), pd.Series(housing.target, name="MedHouseVal")

print("California Housing dataset: "); print(f"Dataset shape: {X.shape}")
print("\nFirst 5 rows: "); print(X.head())
print("\nTarget variable state: "); print(y.describe())

# 2. Visual Analysis (Distribution, Correlation, Missing)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.hist(y, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
ax1.set_title('Distribution of Housing Prices')

X.corrwith(y).sort_values().plot(kind='barh', ax=ax2, color='lightgreen')
ax2.set_title('Feature Correlation with Target')

missing = X.isnull().sum()
if missing.sum() > 0:
    missing.plot(kind='barh', ax=ax3, color='salmon')
else:
    ax3.text(0.5, 0.5, "No missing values", ha='center', fontsize=12)
ax3.set_title('Missing Values Check')

plt.tight_layout(); plt.show()

# 3. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
model = LinearRegression()

model.fit(scaler.fit_transform(X_train), y_train)
y_pred = model.predict(scaler.transform(X_test))

# 4. Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*50 + "\nModel Performance Metrics:\n" + "="*50)
print(f"R2 Score: {r2:.4f}")
print(f"RMSE in Dollars: ${rmse*100000:.2f}")
```

## Prog6 Polynomial Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

# 1. Generate Data
np.random.seed(42)
x, y = make_regression(n_samples=300, n_features=1, noise=10, random_state=42)
y = y + 2*x.flatten()**2 - 3*x.flatten()**3 + np.random.normal(0, 15, len(y)) # Add non-linearity

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Dataset Overview: \nTraining: {x_train.shape[0]}, Test: {x_test.shape[0]}")

# 2. Train & Plot Different Degrees
degrees = [1, 2, 3, 6]
results = {}
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for degree, ax in zip(degrees, axes.flatten()):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    r2, rmse = r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))
    results[degree] = {'r2': r2, 'rmse': rmse}
    
    # Plotting
    x_curve = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
    ax.scatter(x_test, y_test, color='blue', alpha=0.6, label="Actual")
    ax.plot(x_curve, model.predict(x_curve), color="red", lw=2, label="Predicted")
    ax.set_title(f"Degree {degree}\nR2 = {r2:.3f}, RMSE={rmse:.3f}")
    ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()

# 3. Comparison
print("Polynomial reg comparison: ")
for d in degrees:
    print(f"Degree {d}: R2 = {results[d]['r2']:.4f}, RMSE: {results[d]['rmse']:.4f}")

best_degree = max(results, key=lambda k: results[k]['r2'])
print(f"\nBest polynomial degree: {best_degree} (R2= {results[best_degree]['r2']:.4f})")
```
