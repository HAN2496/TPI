import numpy as np
from sklearn.metrics import roc_auc_score


def split_ctx(X, y, frac=0.5):
    """test driver 에피소드를 (context, holdout)으로 분할."""
    s = int(len(y) * frac)
    return (X[:s], y[:s]), (X[s:], y[s:])


def grid(n, step=1):
    """순차 적응 context 크기 그리드: step, 2*step, ..., n (마지막은 항상 포함)."""
    ts = list(range(step, n + 1, step))
    if not ts or ts[-1] < n:
        ts.append(n)
    return ts


class Track:
    """순차 적응 곡선 추적. add(t, probs) -> 새 peak이면 True.

    peak/final 정의를 방법 간에 통일하는 게 목적. 루프 몸통(모델 적응,
    모델 전용 상태 기록)은 스크립트 소유.
    """

    def __init__(self, y):
        self.y = np.asarray(y)
        self.ts, self.aurocs, self.probs = [], [], []
        self.peak_i = -1

    def add(self, t, probs):
        a = float(roc_auc_score(self.y, probs))
        self.ts.append(t)
        self.aurocs.append(a)
        self.probs.append(np.asarray(probs))
        if self.peak_i < 0 or a > self.aurocs[self.peak_i]:
            self.peak_i = len(self.ts) - 1
            return True
        return False

    @property
    def peak(self):     # (t, auroc, probs)
        return self.ts[self.peak_i], self.aurocs[self.peak_i], self.probs[self.peak_i]

    @property
    def final(self):    # (t, auroc, probs)
        return self.ts[-1], self.aurocs[-1], self.probs[-1]

    def __len__(self):
        return len(self.ts)
