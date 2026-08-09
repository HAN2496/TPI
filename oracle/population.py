"""가상 사용자 집단 생성 — GT 축의 유일한 분기점 (synthetic | posterior 승격).

archetypes와 phi는 각 환경 패키지의 users.py가 공급한다 (envs/<env>/users.py).
여기는 환경 무관 메커니즘만: 표준화·정규화·threshold·확률화.
"""
from pathlib import Path

import numpy as np

from .base import Oracle
from .true_reward import LinearReward


def synthetic(bank, phi, archetypes, n_users, seed, theta_sd=0.15,
              beta=1.0, flip=0.0, threshold_quantile=0.5):
    """기준 뱅크 bank로 표준화·threshold를 적합해 n_users명의 Oracle을 만든다."""
    feats = [phi(tau) for tau in bank]
    names = list(feats[0])
    F = np.asarray([[f[n] for n in names] for f in feats])
    mean, sd = F.mean(axis=0), F.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (F - mean) / sd

    rng = np.random.default_rng(seed)
    labels = list(archetypes)
    users = []
    for u in range(n_users):
        archetype = labels[u % len(labels)]
        w = {**dict.fromkeys(names, 0.0), **archetypes[archetype]}
        theta = np.asarray([w[n] for n in names]) + rng.normal(0.0, theta_sd, len(names))
        theta *= np.sqrt(len(theta)) / np.linalg.norm(theta)
        threshold = float(np.quantile(Z @ theta, threshold_quantile))
        reward = LinearReward(dict(zip(names, theta)), phi, mean, sd, threshold)
        users.append(Oracle(reward, beta, flip, name=f"user_{u:03d}_{archetype}"))
    return users


def calibrate(users, bank, quantile=0.5):
    """설계형 User(threshold 속성 보유)의 threshold를 뱅크 기준으로 적합 — 라벨 균형."""
    for u in users:
        u.threshold = 0.0
        u.threshold = float(np.quantile([u.R(tau) for tau in bank], quantile))
    return users


def from_run(run_dir, beta=1.0, flip=0.0):
    """실데이터 추정 posterior를 GT로 승격 — 추정물이 oracle로 건너가는 유일한 문.

    run_dir: model.joblib이 있는 디렉토리 (예: outputs/fully_bayesian/<ts>/true).
    """
    import joblib
    from reward.fully_bayesian.model import Population
    from reward.fully_bayesian.reward_model import PosteriorReward

    obj = joblib.load(Path(run_dir) / "model.joblib")
    phi, pop = obj["phi"], Population.from_state_dict(obj["pop"])
    return [Oracle(PosteriorReward(phi, pop.user(n)), beta, flip, name=n)
            for n in pop.user_names]
