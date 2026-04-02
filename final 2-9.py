import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ===============================
# SKLEARN IMPORTS
# ===============================
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# MODELS
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.neural_network import *

# PCA + CLUSTERING
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import scipy.cluster.hierarchy as shc

# METRICS
from sklearn.metrics import *

# ===============================
# CHANGE EXPERIMENT HERE
# ===============================
EXPERIMENT = 2   # 2 → 9

# ===============================
# LOAD DATASET
# ===============================

if EXPERIMENT == 7:
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

elif EXPERIMENT == 8:
    df = pd.read_csv("pca_dataset.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

elif EXPERIMENT == 9:
    df = pd.read_csv("har.csv")
    X = df.drop(columns=["Activity"])
    y = df["Activity"]

else:
    df = pd.read_csv("spam.csv")
    X = df.drop("class", axis=1)
    y = df["class"]

# ===============================
# PREPROCESSING
# ===============================
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# ===============================
# TRAIN TEST SPLIT
# ===============================
if EXPERIMENT != 9:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# ============================================================
# ======================= EXPERIMENT 2 ========================
# ============================================================
if EXPERIMENT == 2:

    models = {
        "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [3,5,7]}),
        "NaiveBayes": (GaussianNB(), {})
    }

    for name, (model, param_grid) in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        grid = GridSearchCV(pipe, param_grid, cv=5)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)

        print(f"\n{name}")
        print(classification_report(y_test, y_pred))

# ============================================================
# ======================= EXPERIMENT 3 ========================
# ============================================================
elif EXPERIMENT == 3:

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet()
    }

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        print(f"\n{name}")
        print("R2:", r2_score(y_test, y_pred))
        print("MSE:", mean_squared_error(y_test, y_pred))

# ============================================================
# ======================= EXPERIMENT 4 ========================
# ============================================================
elif EXPERIMENT == 4:

    models = {
        "Logistic": LogisticRegression(max_iter=5000),
        "SVM": SVC(probability=True)
    }

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        print(f"\n{name}")
        print(classification_report(y_test, y_pred))

# ============================================================
# ======================= EXPERIMENT 5 ========================
# ============================================================
elif EXPERIMENT == 5:

    models = {
        "Perceptron": Perceptron(),
        "MLP": MLPClassifier(max_iter=1000)
    }

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        print(f"\n{name}")
        print(classification_report(y_test, y_pred))

# ============================================================
# ======================= EXPERIMENT 6 ========================
# ============================================================
elif EXPERIMENT == 6:

    models = {
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        print(f"\n{name}")
        print(classification_report(y_test, y_pred))

# ============================================================
# ======================= EXPERIMENT 7 ========================
# ============================================================
elif EXPERIMENT == 7:

    models = {
        "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100),
        "AdaBoost": AdaBoostClassifier(n_estimators=100),
        "GradientBoost": GradientBoostingClassifier(n_estimators=100),
        "Stacking": StackingClassifier(
            estimators=[
                ("svm", SVC(probability=True)),
                ("nb", GaussianNB()),
                ("dt", DecisionTreeClassifier())
            ],
            final_estimator=LogisticRegression()
        )
    }

    results = []

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append([name, acc, prec, rec, f1])

    print(pd.DataFrame(results,
          columns=["Model","Accuracy","Precision","Recall","F1"]))

# ============================================================
# ======================= EXPERIMENT 8 ========================
# ============================================================
elif EXPERIMENT == 8:

    models = {
        "Logistic": LogisticRegression(max_iter=5000),
        "SVM": SVC()
    }

    results = []

    for name, model in models.items():

        pipe1 = Pipeline([("prep", preprocessor), ("model", model)])
        pipe1.fit(X_train, y_train)
        acc1 = accuracy_score(y_test, pipe1.predict(X_test))

        pipe2 = Pipeline([
            ("prep", preprocessor),
            ("pca", PCA(n_components=0.95)),
            ("model", model)
        ])
        pipe2.fit(X_train, y_train)
        acc2 = accuracy_score(y_test, pipe2.predict(X_test))

        results.append([name, acc1, acc2])

    print(pd.DataFrame(results,
          columns=["Model","Without PCA","With PCA"]))

    # Scree plot
    X_scaled = preprocessor.fit_transform(X)
    pca = PCA().fit(X_scaled)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title("Scree Plot")
    plt.show()

# ============================================================
# ======================= EXPERIMENT 9 ========================
# ============================================================
elif EXPERIMENT == 9:

    X_processed = preprocessor.fit_transform(X)

    models = {
        "KMeans": KMeans(n_clusters=6),
        "DBSCAN": DBSCAN(eps=2.5, min_samples=5),
        "Hierarchical": AgglomerativeClustering(n_clusters=6)
    }

    results = []

    for name, model in models.items():
        labels = model.fit_predict(X_processed)

        if len(set(labels)) > 1:
            sil = silhouette_score(X_processed, labels)
            db = davies_bouldin_score(X_processed, labels)
            ari = adjusted_rand_score(y, labels)
            nmi = normalized_mutual_info_score(y, labels)
        else:
            sil, db, ari, nmi = 0,0,0,0

        results.append([name, sil, db, ari, nmi])

        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X_processed)
        plt.scatter(X_tsne[:,0], X_tsne[:,1], c=labels)
        plt.title(name)
        plt.show()

    print(pd.DataFrame(results,
          columns=["Model","Silhouette","DB","ARI","NMI"]))

    # Elbow
    wcss = []
    sil = []

    for k in range(2,9):
        km = KMeans(n_clusters=k)
        labels = km.fit_predict(X_processed)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X_processed, labels))

    plt.plot(range(2,9), wcss)
    plt.title("Elbow")
    plt.show()

    plt.plot(range(2,9), sil)
    plt.title("Silhouette")
    plt.show()

    plt.figure(figsize=(10,5))
    shc.dendrogram(shc.linkage(X_processed, method='ward'))
    plt.title("Dendrogram")
    plt.show()