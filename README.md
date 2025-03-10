# doc-scikit-learn

# Scikit-Learn Reference Card

## Table of Contents
1. [Installation and Setup](#installation-and-setup)
2. [Data Preparation](#data-preparation)
3. [Feature Engineering](#feature-engineering)
4. [Model Selection and Training](#model-selection-and-training)
5. [Supervised Learning](#supervised-learning)
6. [Unsupervised Learning](#unsupervised-learning)
7. [Model Evaluation](#model-evaluation)
8. [Model Tuning](#model-tuning)
9. [Pipeline API](#pipeline-api)
10. [Common Scikit-learn Patterns](#common-scikit-learn-patterns)

## Installation and Setup

```python
# Standard installation
pip install scikit-learn

# Import commonly used modules
from sklearn import datasets, metrics, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
```

## Data Preparation

### Loading Data
```python
# Built-in datasets
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target

# From pandas
import pandas as pd
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']
```

### Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, GroupKFold, LeaveOneOut

# Basic cross-validation
scores = cross_val_score(model, X, y, cv=5)  # 5-fold CV

# Other CV schemes
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5)  # maintains class distribution
gkf = GroupKFold(n_splits=5)  # for grouped data
loo = LeaveOneOut()  # for small datasets
```

## Feature Engineering

### Scalers
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

# Z-score normalization (μ=0, σ=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scale to range [0,1]
min_max = MinMaxScaler(feature_range=(0, 1))
X_minmax = min_max.fit_transform(X)

# Robust scaling using quartiles (handles outliers better)
robust = RobustScaler()
X_robust = robust.fit_transform(X)

# Normalize samples to unit norm
norm = Normalizer(norm='l2')  # l1, l2, or max
X_norm = norm.fit_transform(X)
```

### Encoders
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Encode target/class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Transforms to [0, 1, 2, ...]
original_labels = le.inverse_transform(y_encoded)

# One-hot encoding (categorical to dummy variables)
ohe = OneHotEncoder(sparse_output=False)
X_cat_encoded = ohe.fit_transform(X_categorical)

# Ordinal encoding (when order matters)
ord_enc = OrdinalEncoder()
X_ordinal = ord_enc.fit_transform(X_categorical)
```

### Imputation
```python
from sklearn.impute import SimpleImputer, KNNImputer

# Fill missing values
imp_mean = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', 'constant'
X_imputed = imp_mean.fit_transform(X)

# KNN imputation
imp_knn = KNNImputer(n_neighbors=5)
X_knn_imputed = imp_knn.fit_transform(X)
```

### Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold

# Select k best features
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

# Recursive feature elimination
from sklearn.linear_model import LogisticRegression
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)

# Remove low-variance features
var_thresh = VarianceThreshold(threshold=0.1)
X_var = var_thresh.fit_transform(X)
```

### Dimensionality Reduction
```python
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

# Principal Component Analysis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

# Truncated SVD (works with sparse matrices)
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

# t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
```

## Model Selection and Training

### The Estimator API
All scikit-learn models follow these methods:
```python
# Estimator initialization
model = Algorithm(param1=val1, param2=val2)

# Fit model to training data
model.fit(X_train, y_train)

# Predict on new data
y_pred = model.predict(X_test)  # For classification/regression
y_proba = model.predict_proba(X_test)  # Probability estimates (for some classifiers)

# Evaluate model
score = model.score(X_test, y_test)  # R² for regression, accuracy for classification
```

### Common Parameters
```python
# Common parameters across estimators:
random_state=42  # For reproducibility
n_jobs=-1  # Use all available cores
verbose=1  # Control output verbosity
```

## Supervised Learning

### Linear Models
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression

# Regression
lr = LinearRegression(fit_intercept=True)
ridge = Ridge(alpha=1.0)  # L2 regularization
lasso = Lasso(alpha=0.1)  # L1 regularization
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # L1 + L2

# Classification
logreg = LogisticRegression(
    C=1.0,  # Inverse of regularization strength
    penalty='l2',  # 'l1', 'l2', 'elasticnet' or None
    solver='lbfgs',  # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    multi_class='auto',  # 'ovr' or 'multinomial'
    class_weight='balanced'  # None, 'balanced', or dict
)
```

### Support Vector Machines
```python
from sklearn.svm import SVC, SVR, LinearSVC

# Classification
svc = SVC(
    C=1.0,  # Regularization parameter
    kernel='rbf',  # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',  # Kernel coefficient
    probability=True  # Enable probability estimates
)

# Regression
svr = SVR(C=1.0, kernel='rbf', gamma='scale')

# Linear SVM (faster for large datasets)
lsvc = LinearSVC(dual='auto')
```

### Tree-Based Models
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)

# Decision Trees
dt_clf = DecisionTreeClassifier(
    max_depth=None,  # Max tree depth
    min_samples_split=2,  # Min samples to split internal node
    min_samples_leaf=1,  # Min samples at leaf node
    criterion='gini'  # 'gini' or 'entropy'
)
dt_reg = DecisionTreeRegressor(criterion='squared_error')

# Random Forest
rf_clf = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=None,
    max_features='sqrt',  # Features to consider: 'sqrt', 'log2', or int
    bootstrap=True,  # Build trees using bootstrap samples
    oob_score=True  # Use out-of-bag samples for estimation
)
rf_reg = RandomForestRegressor(n_estimators=100)

# Gradient Boosting
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0  # Fraction of samples for fitting base learners
)
gb_reg = GradientBoostingRegressor(n_estimators=100)

# AdaBoost
ada_clf = AdaBoostClassifier(n_estimators=50)
ada_reg = AdaBoostRegressor(n_estimators=50)
```

### XGBoost Integration
```python
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Histogram-based Gradient Boosting (similar to XGBoost/LightGBM)
hist_gbc = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=None,  # No maximum depth by default
    early_stopping=True,
    validation_fraction=0.1  # For early stopping
)
hist_gbr = HistGradientBoostingRegressor(max_iter=100)
```

### Neighbors
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Classification
knn_clf = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',  # 'uniform' or 'distance'
    algorithm='auto',  # 'ball_tree', 'kd_tree', 'brute'
    metric='minkowski'  # distance metric
)

# Regression
knn_reg = KNeighborsRegressor(n_neighbors=5)
```

### Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# For continuous data
gnb = GaussianNB()

# For discrete counts (e.g., text)
mnb = MultinomialNB(alpha=1.0)  # Smoothing parameter

# For binary/boolean features
bnb = BernoulliNB(alpha=1.0)
```

### Multi-Layer Perceptron
```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Classification
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100,),  # Tuple with layer sizes
    activation='relu',  # 'identity', 'logistic', 'tanh', 'relu'
    solver='adam',  # 'lbfgs', 'sgd', 'adam'
    alpha=0.0001,  # L2 regularization
    batch_size='auto',
    learning_rate='constant',  # 'constant', 'invscaling', 'adaptive'
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True
)

# Regression
mlp_reg = MLPRegressor(hidden_layer_sizes=(100,))
```

## Unsupervised Learning

### Clustering
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering

# K-Means
kmeans = KMeans(
    n_clusters=8,
    init='k-means++',  # 'random' or array
    n_init=10,  # Number of initializations
    max_iter=300
)
clusters = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# DBSCAN
dbscan = DBSCAN(
    eps=0.5,  # Max distance between samples
    min_samples=5,  # Min samples in neighborhood to be core point
    metric='euclidean'
)
labels = dbscan.fit_predict(X)

# Hierarchical Clustering
agg_clust = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'  # 'ward', 'complete', 'average', 'single'
)
clusters = agg_clust.fit_predict(X)

# Spectral Clustering
spec_clust = SpectralClustering(
    n_clusters=3,
    affinity='nearest_neighbors'  # 'nearest_neighbors', 'rbf', or affinity matrix
)
clusters = spec_clust.fit_predict(X)
```

### Anomaly/Outlier Detection
```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # Expected proportion of outliers
    random_state=42
)
preds = iso_forest.fit_predict(X)  # 1: inlier, -1: outlier

# Local Outlier Factor
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1
)
preds = lof.fit_predict(X)

# Elliptic Envelope (assumes Gaussian distribution)
ee = EllipticEnvelope(contamination=0.1)
preds = ee.fit_predict(X)
```

## Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Detailed report
report = classification_report(y_true, y_pred)
print(report)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# AUC and ROC curve
y_scores = classifier.predict_proba(X_test)[:, 1]  # For binary classification
auc = roc_auc_score(y_true, y_scores)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
avg_precision = average_precision_score(y_true, y_scores)
```

### Regression Metrics
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, median_absolute_error,
    r2_score, explained_variance_score, max_error
)

# Error metrics
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)  # Root MSE
mae = mean_absolute_error(y_true, y_pred)
median_ae = median_absolute_error(y_true, y_pred)
max_err = max_error(y_true, y_pred)

# Variance metrics
r2 = r2_score(y_true, y_pred)
exp_var = explained_variance_score(y_true, y_pred)
```

### Clustering Metrics
```python
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, adjusted_mutual_info_score
)

# Internal metrics (no ground truth)
silhouette = silhouette_score(X, labels)
calinski = calinski_harabasz_score(X, labels)
davies = davies_bouldin_score(X, labels)

# External metrics (with ground truth)
ari = adjusted_rand_score(y_true, labels)
ami = adjusted_mutual_info_score(y_true, labels)
```

## Model Tuning

### Grid Search
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search (exhaustive)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',  # 'precision', 'recall', 'f1', 'roc_auc', etc.
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)
grid_search.fit(X_train, y_train)

# Best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_

# All results
results = grid_search.cv_results_
```

### Randomized Search
```python
from scipy.stats import uniform, randint

# Randomized Search (sampling)
param_dist = {
    'C': uniform(0.1, 100),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': uniform(0.001, 1),
    'degree': randint(1, 5)  # for poly kernel
}

random_search = RandomizedSearchCV(
    SVC(),
    param_distributions=param_dist,
    n_iter=100,  # Number of samples
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
```

## Pipeline API

### Creating Pipelines
```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

# Basic sequential pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('classifier', LogisticRegression())
])

# Simplified pipeline creation
pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=5),
    LogisticRegression()
)

# Mixed-type data pipeline
numeric_features = ['age', 'income', 'score']
categorical_features = ['gender', 'city', 'education']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Using the pipeline
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)

# Pipeline with parameter tuning
param_grid = {
    'preprocessor__num__with_mean': [True, False],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### Feature Union
```python
from sklearn.pipeline import FeatureUnion

# Combine different feature extraction methods
feature_union = FeatureUnion([
    ('pca', PCA(n_components=3)),
    ('select_best', SelectKBest(k=5))
])

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('features', feature_union),
    ('classifier', LogisticRegression())
])
```

## Common Scikit-learn Patterns

### Model Persistence
```python
from sklearn.externals import joblib
import pickle

# Save model to file
joblib.dump(model, 'model.joblib')  # Efficient for large NumPy arrays
pickle.dump(model, open('model.pkl', 'wb'))

# Load model from file
model = joblib.load('model.joblib')
model = pickle.load(open('model.pkl', 'rb'))
```

### Custom Transformers
```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param1=1, param2=2):
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X, y=None):
        # Compute parameters needed for transform
        return self
    
    def transform(self, X):
        # Apply transformation
        return X_transformed
```

### Learning Curves
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, valid_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy'
)

# Calculate mean and standard deviation
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, valid_mean, label='Validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1)
plt.xlabel('Training set size')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
```

### Working with Imbalanced Data
```python
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Compute class weights for imbalanced datasets
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Use class weights in model
model = RandomForestClassifier(class_weight=weight_dict)

# SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Under-sampling
under_sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = under_sampler.fit_resample(X, y)

# Imbalanced-learn pipeline
imb_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('sampling', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier())
])
```

### Multioutput Models
```python
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

# Multioutput classification
multi_clf = MultiOutputClassifier(RandomForestClassifier())
multi_clf.fit(X, y_multioutput)  # y has shape (n_samples, n_targets)

# Multioutput regression
multi_reg = MultiOutputRegressor(RandomForestRegressor())
multi_reg.fit(X, y_multioutput)
```
