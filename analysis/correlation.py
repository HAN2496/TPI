import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mutual_info_score
from scipy.stats import pointbiserialr, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.utils.data_loader import DatasetManager
from src.utils import ExperimentLogger


def extract_data(name):
    manager = DatasetManager("datasets")
    dataset = manager.get(name)
    features = []
    labels = []
    for item in dataset:
        if item['label'] is None:
            continue
        states = item['states']
        segment = states[(states['Time'] >= 5) & (states['Time'] <= 8)]
        numeric_cols = segment.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'Time']
        feat = {}
        for col in numeric_cols:
            v = segment[col].values
            feat[f'{col}_rmse'] = np.sqrt(np.mean((v - v.mean()) ** 2))
            feat[f'{col}_max'] = np.abs(v).max()
        features.append(feat)
        labels.append(1 if item['label'] else 0)
    df = pd.DataFrame(features)
    df = df.dropna(axis=1)
    labels = np.array(labels)
    return df, labels


def test_corr_ttest(name):
    df, labels = extract_data(name)
    results = []
    for col in df.columns:
        corr, pval = pointbiserialr(labels, df[col])
        true_vals = df[col][labels == 1]
        false_vals = df[col][labels == 0]
        t_stat, t_pval = ttest_ind(true_vals, false_vals)
        results.append({
            'feature': col,
            'correlation': corr,
            'p_value': pval,
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'mean_true': true_vals.mean(),
            'mean_false': false_vals.mean()
        })
    results_df = pd.DataFrame(results).sort_values('p_value')
    print("="*105)
    print("Correlation and T-test Results")
    print(results_df.head(20))
    print("="*105)

def test_mutual_info(name):
    df, labels = extract_data(name)
    X = df.values
    y = labels
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    mi_df = pd.DataFrame({'feature': df.columns, 'mi': mi}).sort_values('mi', ascending=False)
    print("="*50)
    print("Mutual Information Results")
    print(mi_df.head(20))
    print("="*50)

def test_mutual_info(name):
    df, labels = extract_data(name)
    X = df.values
    y = labels

    # 원래 MI
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)

    # 라벨 엔트로피 H(Y)
    H_y = mutual_info_score(y, y)  # I(y; y) = H(y)
    # 정규화된 MI (I(X_j; Y) / H(Y))
    nmi = mi / H_y if H_y > 0 else np.zeros_like(mi)
    # nmi = normalized_mutual_info_score
    mi_df = pd.DataFrame({
        'feature': df.columns,
        'mi': mi,
        'nmi': nmi
    }).sort_values('mi', ascending=False)

    print("="*50)
    print("Mutual Information Results")
    print(mi_df.head(20))
    print("="*50)

def test_random_forest_importance(name):
    df, labels = extract_data(name)
    X = df.values
    y = labels
    rf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight='balanced')
    rf.fit(X, y)
    imp = rf.feature_importances_
    imp_df = pd.DataFrame({'feature': df.columns, 'importance': imp}).sort_values('importance', ascending=False)
    print("="*50)
    print("Random Forest Feature Importance")
    print(imp_df.head(20))
    print("="*50)


def test_gradient_boosting_importance(name):
    df, labels = extract_data(name)
    X = df.values
    y = labels
    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(X, y)
    imp = gb.feature_importances_
    imp_df = pd.DataFrame({'feature': df.columns, 'importance': imp}).sort_values('importance', ascending=False)
    print("="*50)
    print("Gradient Boosting Feature Importance")
    print(imp_df.head(20))
    print("="*50)


def test_permutation_importance(name):
    print("="*50)
    print("Permutation Feature Importance")
    df, labels = extract_data(name)
    X = df.values
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    model = RandomForestClassifier(n_estimators=300, random_state=0, class_weight='balanced')
    model.fit(X_train, y_train)
    r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0, scoring='roc_auc')
    imp_df = pd.DataFrame({'feature': df.columns, 'importance': r.importances_mean}).sort_values('importance', ascending=False)
    print(imp_df.head(20))


if __name__ == "__main__":
    for name in ["강신길", "박재일", "한규택"]:
        correlation_dir = Path("artifacts") / name / "correlation"
        correlation_dir.mkdir(parents=True, exist_ok=True)

        logger = ExperimentLogger(str(correlation_dir), "correlation_results", add_timestamp=False)
        logger.start()

        print(f"========================== Correlation Analysis for {name} ==========================")
        test_corr_ttest(name)
        test_mutual_info(name)
        test_random_forest_importance(name)
        test_gradient_boosting_importance(name)
        test_permutation_importance(name)
        print(f"========================== End of Analysis for {name} ==========================")

        logger.stop()
        print(f"\nCorrelation analysis results saved to {correlation_dir / 'correlation_results.txt'}")