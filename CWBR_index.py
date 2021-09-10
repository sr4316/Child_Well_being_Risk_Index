from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def make_index(df):

    '''
    This function performs Principal Components Analysis on the eight indicators that will compose our index.
    The number of principal components will be 2.
    
    Taking df as an input, the function will return four inputs, which are the following:
    
    pipe:
        all combined together
    
    PC_df:
        weights in principal component for each indicator,
    
    X_proj:
        the projected data set
        
    Variance_df:
        percentages captured by each principal component.

    '''
    X = df.dropna()

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2))
    ]).fit(X)

    PC_df = pd.DataFrame(pipe['pca'].components_.T, columns=['PC_1', 'PC_2'], index=X.columns)

    Variance_df = pd.DataFrame([pipe['pca'].explained_variance_ratio_], columns=['PC_1', 'PC_2'])

    X_proj = pd.DataFrame(pipe.transform(X), columns=['PC_1', 'PC_2'], index=X.index)

    return pipe, PC_df, X_proj, Variance_df