from dataclasses import dataclass, field
import time
import numpy as np
import torch

from loader import Dataset
from core import Run
from reward.reconstruction import methods, viz
from reward.reconstruction.config import KalmanConfig, LSTMConfig
from reward.reconstruction.methods import physics, Kalman, FIR, LSTM, UNet
from reward.fully_bayesian.features import FNS


@dataclass
class Config:
    # run_fully_bayesian.Config.test와 반드시 동일하게 유지 (재구성기는 test driver를 절대 학습에 쓰지 않는다).
    # train pool = test 이외 전부 (분류기 train 5명 + 미지정 driver 포함, 라벨 불필요라 전량 사용).
    test: tuple = ("김재호", "김진명", "김태근", "신민철", "이강근")

    # 배포 가용 신호 (회사 확인): 거동 = IMU + WHL만, 토크 입력 사용 가능. SAS/페달/브레이크 불가.
    # physics·Kalman이 이 순서를 전제로 위치 인덱싱 (IMU 5, WHL 4, 토크 4) — 순서 변경 금지.
    x_channels: tuple = ("IMU_RollRtVal", "IMU_VerAccelVal", "IMU_YawRtVal", "IMU_LatAccelVal", "IMU_LongAccelVal",
                         "WHL_SpdFLVal", "WHL_SpdFRVal", "WHL_SpdRLVal", "WHL_SpdRRVal",
                         "MCU_Mg1EstTqVal", "MCU_Mg2EstTqVal", "VCU_MotTqCmdFrntVal", "VCU_MotTqCmdRearVal")
    y_channels: tuple = ("Bounce_rate_6D", "Roll_rate_6D", "Pitch_rate_6D")
    fs: float = 100.0
    episode_len: int = 1000
    device: str = "cuda"

    timestamp: str = None
    seed: int = 42

    fir_epochs: int = 20
    # Driver-held-out sweep: 42,915 params; selected by the performance-size Pareto rule.
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    unet_epochs: int = 40
    methods: tuple = ("physics", "kalman", "fir", "unet")   # 사용 가능: physics/kalman/fir/lstm/unet
    show: tuple = ("physics", "kalman", "fir", "unet")      # methods의 부분집합, 마지막이 시각화 기준


def load(ds, cfg):
    xs, ys, ids = [], [], []
    for ep in ds.episodes:
        df = ep.signals.df
        x = df[list(cfg.x_channels)].to_numpy(np.float32)
        y = df[list(cfg.y_channels)].to_numpy(np.float32)
        if len(x) < cfg.episode_len:
            x = np.pad(x, ((0, cfg.episode_len - len(x)), (0, 0)), mode="edge")
            y = np.pad(y, ((0, cfg.episode_len - len(y)), (0, 0)), mode="edge")
        xs.append(x[:cfg.episode_len])
        ys.append(y[:cfg.episode_len])
        ids.append(f"{ep.driver} {ep.id}")
    return np.stack(xs), np.stack(ys), np.array(ids)


def main(cfg=None, run=None):
    # run 지정 시 외부 run 폴더에 산출물 저장 (run_fully_bayesian의 recon.timestamp=None 인라인 학습용)
    cfg = cfg or Config()
    available = {"physics", "kalman", "fir", "lstm", "unet"}
    unknown = set(cfg.methods) - available
    if unknown:
        raise ValueError(f"unknown reconstruction methods: {sorted(unknown)}")
    if set(cfg.show) - set(cfg.methods):
        raise ValueError(f"show에 피팅하지 않는 method 포함: {sorted(set(cfg.show) - set(cfg.methods))}")
    standalone = run is None
    if standalone:
        run = Run("reconstruction", cfg)
    run.plots.mkdir(parents=True, exist_ok=True)
    xr, yr, ids = load(Dataset("datasets"), cfg)
    te = np.array([i.split()[0] in cfg.test for i in ids])
    tt = torch.tensor(te)
    print(f"[INFO] reconstruction  {len(xr)} episodes  train {(~te).sum()}  test {te.sum()}")

    tic = time.time()
    ph = np.stack([physics(x, cfg.fs) for x in xr])
    stat = lambda a: (a[~te].reshape(-1, a.shape[2]).mean(0), a[~te].reshape(-1, a.shape[2]).std(0) + 1e-8)
    (mx, sx), (my, sy), (mp, sp) = stat(xr), stat(yr), stat(ph)
    to = lambda a, m, s: torch.tensor(((a - m) / s).transpose(0, 2, 1))
    x, y, p = to(xr, mx, sx), to(yr, my, sy), to(ph, mp, sp)
    print(f"Load + Physics Done. ({time.time() - tic:.2f} sec)")

    preds = {}
    ch = len(cfg.x_channels) + len(cfg.y_channels)
    state = torch.load(run.dir / "models.pt", weights_only=False) if run.eval_only else {}
    if "physics" in cfg.methods:
        preds["physics"] = p.numpy()
    if "kalman" in cfg.methods:
        if run.eval_only:
            km = Kalman(state["kalman"], cfg.fs, cfg.kalman.warmup)
        else:
            print("kalman")
            km = Kalman.fit(xr[~te], yr[~te], cfg.fs, cfg.kalman.warmup, cfg.kalman.eta_t)
            state["kalman"] = km.p
        kal = km.predict(xr)
        mk, sk = kal[~te].mean((0, 2), keepdims=True), kal[~te].std((0, 2), keepdims=True)
        preds["kalman"] = (kal - mk) / sk
    model_specs = (
        ("fir", lambda: FIR(ch), cfg.fir_epochs, True, {}),
        ("lstm", lambda: LSTM(len(cfg.x_channels), len(cfg.y_channels),
                              **cfg.lstm.model_kwargs()),
         cfg.lstm.epochs, False, cfg.lstm.fit_kwargs()),
        ("unet", lambda: UNet(ch), cfg.unet_epochs, True, {}),
    )
    for name, make_model, epochs, residual, fit_kwargs in model_specs:
        if name not in cfg.methods:
            continue
        net = make_model()
        if run.eval_only:
            net.load_state_dict(state[name])
        else:
            n_params = sum(v.numel() for v in net.parameters())
            print(f"{name}  params={n_params:,}")
            methods.fit(net, x[~tt], y[~tt], p[~tt] if residual else None,
                        epochs, cfg.device, **fit_kwargs)
            state[name] = net.state_dict()
        preds[name] = methods.predict(net, x, p if residual else None, cfg.device)
    if not run.eval_only:
        torch.save(state, run.dir / "models.pt")

    # 평가는 test driver만
    yte = y.numpy()[te]
    preds_te = {m: pr[te] for m, pr in preds.items()}
    R = {m: viz.waveform_r(pr, yte) for m, pr in preds_te.items()}
    print(f"\n{'':8s}" + "".join(f"{c:>28s}" for c in cfg.y_channels))
    for m, r in R.items():
        med, lo, hi = np.nanmedian(r, 0), np.nanpercentile(r, 25, 0), np.nanpercentile(r, 75, 0)
        print(f"{m:8s}" + "".join(f"     {a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in zip(med, lo, hi)))
        run.metrics[f"test/{m}"] = {c: float(v) for c, v in zip(cfg.y_channels, med)}

    print(f"\nbounce stats corr (test episodes)\n{'':8s}" + "".join(f"{s:>16s}" for s in viz.BOUNCE_STATS))
    for m, pr in preds_te.items():
        rs = {s: float(viz.corr(FNS[s](yte[:, 0], cfg.fs), FNS[s](pr[:, 0], cfg.fs)))
              for s in viz.BOUNCE_STATS}
        print(f"{m:8s}" + "".join(f"{v:16.3f}" for v in rs.values()))
        run.metrics[f"test/{m}"].update(rs)

    viz.plot_overlay(yte, preds_te, R, ids[te], run.plots / "overlay.png", cfg.show, cfg.y_channels, cfg.fs)
    viz.plot_r_distribution(R, run.plots / "waveform_correlation_distribution.png", cfg.show, cfg.y_channels)
    viz.plot_stats_scatter(yte, preds_te, run.plots / "stats_scatter.png", cfg.show, cfg.fs)

    # 다운스트림용 아티팩트: 전 에피소드의 재구성 신호 (native 단위, y 통계로 affine 보정)
    native = {m: (pr * sy[:, None] + my[:, None]).astype(np.float32) for m, pr in preds.items()}
    np.savez(run.dir / "reconstructed_signals.npz", ids=ids, te=te, channels=np.array(cfg.y_channels),
             y=yr.transpose(0, 2, 1), **native)
    print(f"\nartifact: reconstructed_signals.npz ({len(xr)} episodes x {len(preds)} methods)")
    if standalone:
        run.finish()


if __name__ == "__main__":
    main()
