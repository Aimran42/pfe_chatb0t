import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.title("Classification de la Qualité du Phosphate")

# 1 Importation du Dataset

uploaded_file = st.file_uploader("Importer votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Aperçu du Dataset")
    st.write(df.head())

    st.write("Nombre de lignes :", df.shape[0])
    st.write("Nombre de colonnes :", df.shape[1])

    st.subheader("Types des variables")
    st.write(df.dtypes)

    st.subheader("Valeurs manquantes")
    st.write(df.isnull().sum())

    st.subheader("Statistiques descriptives")
    st.write(df.describe())

# 2 Visualisations

    st.subheader("Distribution des classes")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Qualite_Phosphate", data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Histogrammes")
    df.hist(figsize=(10, 8))
    st.pyplot(plt.gcf())

    st.subheader("Matrice de corrélation")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Boxplots")
    for col in df.select_dtypes(include=np.number).columns:
        fig, ax = plt.subplots()
        sns.boxplot(x="Qualite_Phosphate", y=col, data=df, ax=ax)
        st.pyplot(fig)

#Prétraitement
    X = df.drop("Qualite_Phosphate", axis=1)
    y = df["Qualite_Phosphate"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

#  Modèles

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }

    results = []

    st.subheader("Évaluation des Modèles")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        results.append([name, acc, prec, rec, f1])

        st.write(f"### {name}")
        st.write("Accuracy:", acc)
        st.write("Precision:", prec)
        st.write("Recall:", rec)
        st.write("F1-score:", f1)

        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

    # Comparaison

    st.subheader("Comparaison des Modèles")
    results_df = pd.DataFrame(results, columns=["Modèle", "Accuracy", "Precision", "Recall", "F1-score"])
    st.write(results_df)

    # Prédiction manuelle

    st.subheader("Prédiction d'un nouveau phosphate")

    p2o5 = st.number_input("P2O5 (%)")
    cao = st.number_input("CaO (%)")
    sio2 = st.number_input("SiO2 (%)")
    mgo = st.number_input("MgO (%)")
    humidite = st.number_input("Humidité (%)")

    if st.button("Prédire"):
        input_data = np.array([[p2o5, cao, sio2, mgo, humidite]])
        input_data = imputer.transform(input_data)
        input_data = scaler.transform(input_data)

        best_model = RandomForestClassifier()
        best_model.fit(X_train, y_train)
        prediction = best_model.predict(input_data)

        st.success("Qualité prédite : " + le.inverse_transform(prediction)[0])