import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin

def target_encode(train, target_col, exclude_cols=['기준년월', 'ID'], n_splits=5, smoothing=10):
    train = train.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 범주형 변수 자동 감지 (object, category)
    cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [col for col in cat_cols if col not in exclude_cols]

    global_mean = train[target_col].mean()

    for col in cat_cols:
        print(f'Encoding {col}')
        train[f'{col}_te'] = np.nan

        for train_idx, val_idx in kf.split(train):
            train_fold = train.iloc[train_idx]

            stats = train_fold.groupby(col)[target_col].agg(['mean', 'count'])
            means = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)

            # 인코딩 적용
            train.loc[val_idx, f'{col}_te'] = train.loc[val_idx, col].map(means)

        # 만약 매핑이 안 된 값이 있을 경우만 대비해서 (극히 드물게)
        train[f'{col}_te'] = train[f'{col}_te'].fillna(global_mean)
        # 원래 변수 제거
        train.drop(columns=[col], inplace=True)

    return train


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        
    def fit(self, X, y):
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X if str(X[col].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.maps = dict() #dict to store map for each column
        for col in self.cols:
            tmap = dict()
            uniques = X[col].unique()
            for unique in uniques:
                tmap[unique] = y[X[col]==unique].mean()
            self.maps[col] = tmap
            
        return self

        
    def transform(self, X, y=None):
        Xo = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan)
            for val, mean_target in tmap.items():
                vals[X[col]==val] = mean_target
            Xo[col] = vals
        return Xo
            
            
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)