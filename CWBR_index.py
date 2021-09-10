from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def make_index(df):
    X = df.dropna()

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2))
    ]).fit(X)

    PC_df = pd.DataFrame(pipe['pca'].components_.T, columns=['PC_1', 'PC_2'], index=X.columns)

    Variance_df = pd.DataFrame([pipe['pca'].explained_variance_ratio_], columns=['PC_1', 'PC_2'])

    X_proj = pd.DataFrame(pipe.transform(X), columns=['PC_1', 'PC_2'], index=X.index)

    return pipe, PC_df, X_proj, Variance_df