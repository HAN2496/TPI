import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters

from .utils import to_tsfresh_df

class TSFreshPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, n_components=2, extraction_settings=None, verbose=0):
        self.feature_names = feature_names
        self.n_components = n_components
        self.verbose = verbose
        
        # Explicit settings logic
        if extraction_settings == 'minimal':
            self.settings = MinimalFCParameters()
        elif extraction_settings == 'efficient' or extraction_settings is None:
            self.settings = EfficientFCParameters()
        else:
            self.settings = extraction_settings

        self.selector = VarianceThreshold(threshold=0.0)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        
        self.input_columns_ = None
        self.selected_columns_ = None
        self.fitted_ = False

    def _extract(self, X):
        # No defensive checks, let it crash if X is wrong
        df_long = to_tsfresh_df(X, self.feature_names)
        
        if self.verbose > 0:
            print(f"[TSFresh] Extracting features from shape {X.shape}...")
            
        X_extracted = extract_features(
            df_long, 
            column_id="id", 
            column_sort="time", 
            default_fc_parameters=self.settings,
            n_jobs=0, 
            disable_progressbar=(self.verbose < 1)
        )
        
        impute(X_extracted)
        return X_extracted

    def fit(self, X, y=None):
        X_extracted = self._extract(X)
        
        # 1. Variance Threshold
        self.selector.fit(X_extracted)
        X_sel = self.selector.transform(X_extracted)
        
        self.input_columns_ = X_extracted.columns
        self.selected_columns_ = X_extracted.columns[self.selector.get_support()]
        
        if self.verbose > 0:
            print(f"[TSFresh] Features: {X_extracted.shape[1]} -> {X_sel.shape[1]} (VarianceThreshold)")

        # 2. Scale
        self.scaler.fit(X_sel)
        X_scaled = self.scaler.transform(X_sel)
        
        # 3. PCA
        self.pca.fit(X_scaled)
        
        if self.verbose > 0:
            expl = np.sum(self.pca.explained_variance_ratio_)
            print(f"[TSFresh] PCA Fitted. Explained Variance ({self.n_components} comps): {expl:.4f}")
        
        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Pipeline not fitted yet.")
            
        X_extracted = self._extract(X)
        
        # Strict alignment: Reindex using training columns.
        # This will introduce NaNs if new columns appear (unlikely with same settings)
        # or fill missing ones with 0.
        X_aligned = X_extracted.reindex(columns=self.input_columns_, fill_value=0.0)
        
        X_sel = self.selector.transform(X_aligned)
        X_scaled = self.scaler.transform(X_sel)
        Z = self.pca.transform(X_scaled)
        
        return Z
    
    def save(self, path):
        joblib.dump(self, path)
        
    @staticmethod
    def load(path):
        return joblib.load(path)