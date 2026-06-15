import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier


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
