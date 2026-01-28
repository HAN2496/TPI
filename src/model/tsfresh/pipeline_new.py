import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters, ComprehensiveFCParameters

from .utils import to_tsfresh_df

class TSFreshPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, n_components=2, extraction_settings=None, verbose=0):
        self.feature_names = feature_names
        self.n_components = n_components
        self.verbose = verbose
        
        # Settings logic
        if extraction_settings == 'minimal':
            self.settings = MinimalFCParameters()
        elif extraction_settings == 'comprehensive': # <-- 추가된 부분
            self.settings = ComprehensiveFCParameters()
        elif extraction_settings == 'efficient' or extraction_settings is None:
            self.settings = EfficientFCParameters()
        else:
            # 사용자가 직접 정의한 딕셔너리(Custom Settings)가 들어오는 경우
            self.settings = extraction_settings
        self.selector = VarianceThreshold(threshold=0.0)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        
        self.input_columns_ = None
        self.fitted_ = False

    def _extract(self, X):
        """Raw Time-Series -> TSFresh Features"""
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

    def _make_difference_vectors(self, X_scaled, y):
        """
        [DRM Paper Implementation]
        Generates difference vectors: z = phi(x_good) - phi(x_bad)
        """
        X_pos = X_scaled[y == 1]
        X_neg = X_scaled[y == 0]

        if len(X_pos) == 0 or len(X_neg) == 0:
            raise ValueError("Data must contain both labels (0 and 1) to compute preference differences.")

        # Create pairs: Since we don't have explicit pairs, we randomly sample
        # to create a robust dataset of "Good - Bad" directions.
        # We aim for roughly N samples to match original data size scale.
        n_pairs = max(len(X_scaled), 1000) 
        
        # Random sampling with replacement
        idx_pos = np.random.choice(len(X_pos), n_pairs)
        idx_neg = np.random.choice(len(X_neg), n_pairs)
        
        Z_diff = X_pos[idx_pos] - X_neg[idx_neg]
        
        if self.verbose > 0:
            print(f"[DRM] Generated {len(Z_diff)} preference difference vectors for PCA.")
            
        return Z_diff

    def fit(self, X, y=None, use_preference_diff=True):
        """
        fit performs: Extract -> Select -> Scale -> (Make Diff) -> PCA
        """
        # 1. Extract Features
        X_extracted = self._extract(X)
        
        # 2. Variance Threshold
        self.selector.fit(X_extracted)
        X_sel = self.selector.transform(X_extracted)
        
        self.input_columns_ = X_extracted.columns
        # Keep track of columns for re-indexing during transform
        self.selected_columns_ = X_extracted.columns[self.selector.get_support()]
        
        if self.verbose > 0:
            print(f"[Pipeline] Feature Selection: {X_extracted.shape[1]} -> {X_sel.shape[1]}")

        # 3. Scale
        self.scaler.fit(X_sel)
        X_scaled = self.scaler.transform(X_sel)
        
        # 4. PCA (The Key Step)
        if use_preference_diff and y is not None:
            # DRM Mode: Fit PCA on (Good - Bad) vectors
            # This finds the "axes of preference"
            X_for_pca = self._make_difference_vectors(X_scaled, y)
        else:
            # Standard Mode: Fit PCA on raw variations
            X_for_pca = X_scaled
            
        self.pca.fit(X_for_pca)
        
        if self.verbose > 0:
            expl = np.sum(self.pca.explained_variance_ratio_)
            mode_str = "Preference Differences" if use_preference_diff else "Raw Data"
            print(f"[Pipeline] PCA Fitted on {mode_str}. Explained Variance: {expl:.4f}")
        
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Transforms new data into the learned Basis space.
        Note: We project the raw feature vector phi(x), NOT the difference.
        The user adaptation (Logistic Regression) will handle the weighting.
        """
        if not self.fitted_:
            raise RuntimeError("Pipeline not fitted yet.")
            
        X_extracted = self._extract(X)
        
        # Strict alignment
        X_aligned = X_extracted.reindex(columns=self.input_columns_, fill_value=0.0)
        
        X_sel = self.selector.transform(X_aligned)
        X_scaled = self.scaler.transform(X_sel)
        
        # Project data onto the basis (W) learned from differences
        Z = self.pca.transform(X_scaled)
        
        return Z
    
    def save(self, path):
        joblib.dump(self, path)
        
    @staticmethod
    def load(path):
        return joblib.load(path)