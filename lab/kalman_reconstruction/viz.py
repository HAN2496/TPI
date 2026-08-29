import matplotlib.pyplot as plt
import numpy as np

from .state_space import highpass

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 15,
})

DISPLAY_NAMES = {"rw": "KF", "oscillator": "2-state KF", "ou": "KF", "rw_lstm": "KF-LSTM",
                 "ou_lstm": "KF-LSTM",
                 "matern32": "KF", "matern32_lstm": "KF-LSTM",
                 "kinematic_kf": "Kinematic KF", "imu_lstm": "LSTM",
                 "lstm_kf": "LSTM-KF", "pitch_hc": "Pitch KF", "pitch_hc_ou": "Pitch KF + OU",
                 "pitch_hc_osc": "Pitch KF + 2nd-order", "pitch_road": "Pitch KF + road",
                 "pitch_road_osc": "Pitch KF + road + 2nd", "pitch_delay": "Pitch KF + delay road",
                 "pitch_delay_osc": "Pitch KF + delay road + 2nd", "pitch_tq": "Pitch KF + torque",
                 "pitch_ax": "Pitch KF + ax obs", "pitch_axou": "Pitch KF + ax + OU road",
                 "pitch_eps": "Pitch KF + split damping"}
LINE_STYLES = {
    "rw": {"color": "#0000FF", "linestyle": "--", "lw": 1.1},
    "rw_lstm": {"color": "#FF0000", "linestyle": "-", "lw": 1.1},
    "oscillator": {"color": "#008000", "linestyle": "-.", "lw": 1.0},
    "ou": {"color": "#0000FF", "linestyle": "--", "lw": 1.1},
    "ou_lstm": {"color": "#FF0000", "linestyle": "-", "lw": 1.1},
    "matern32": {"color": "#0000FF", "linestyle": "--", "lw": 1.1},
    "matern32_lstm": {"color": "#FF0000", "linestyle": "-", "lw": 1.1},
    "kinematic_kf": {"color": "#0000FF", "linestyle": "--", "lw": 1.0},
    "imu_lstm": {"color": "#008000", "linestyle": "-.", "lw": 1.0},
    "lstm_kf": {"color": "#FF0000", "linestyle": "-", "lw": 1.2},
    "pitch_hc": {"color": "#0000FF", "linestyle": "--", "lw": 1.1},
    "pitch_hc_ou": {"color": "#008000", "linestyle": "-.", "lw": 1.0},
    "pitch_hc_osc": {"color": "#7b4ab5", "linestyle": "-", "lw": 1.0},
    "pitch_road": {"color": "#e67e22", "linestyle": "--", "lw": 1.0},
    "pitch_road_osc": {"color": "#d14a8c", "linestyle": "-", "lw": 1.0},
    "pitch_delay": {"color": "#00a1a7", "linestyle": "--", "lw": 1.0},
    "pitch_delay_osc": {"color": "#FF0000", "linestyle": "-", "lw": 1.2},
    "pitch_tq": {"color": "#7b4ab5", "linestyle": "-.", "lw": 1.0},
    "pitch_ax": {"color": "#e67e22", "linestyle": "--", "lw": 1.0},
    "pitch_axou": {"color": "#008000", "linestyle": "-.", "lw": 1.0},
    "pitch_eps": {"color": "#FF0000", "linestyle": "-", "lw": 1.2},
}
MODEL_FREE_DISPLAY_NAMES = {
    "lstm_online": "LSTM",
    "gru_online": "GRU",
    "transformer_online": "Transformer",
    "bilstm_offline": "Bi-LSTM",
    "transformer_offline": "Offline Transformer",
    "unet_offline": "1-D U-Net",
}
MODEL_FREE_LINE_STYLES = {
    "lstm_online": {"color": "#0000FF", "linestyle": "--", "lw": 1.0, "zorder": 2},
    "gru_online": {"color": "#FF0000", "linestyle": "-", "lw": 1.2, "zorder": 4},
    "transformer_online": {"color": "#008000", "linestyle": "-.", "lw": 1.0, "zorder": 3},
    "bilstm_offline": {"color": "#FF0000", "linestyle": "-", "lw": 1.2, "zorder": 4},
    "transformer_offline": {"color": "#0000FF", "linestyle": "--", "lw": 1.0, "zorder": 2},
    "unet_offline": {"color": "#008000", "linestyle": "-.", "lw": 1.0, "zorder": 3},
}
MODEL_FREE_PLOT_ORDER = {
    "online": ("lstm_online", "gru_online", "transformer_online"),
    "offline": ("bilstm_offline", "transformer_offline"),
}


def plot_waveforms(true, results, ids, fs, path, reference):
    order = np.argsort(results[reference]["corr"])
    picks = (order[0], order[len(order) // 2], order[-1])
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    time = np.arange(true.shape[1]) / fs
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, label, index in zip(axes.flat[:3], ("worst", "median", "best"), picks):
        ax.plot(time, true[index], color="black", lw=1.1, label="recorded")
        for color, (name, result) in zip(colors, results.items()):
            style = {"color": color, "lw": .9} | LINE_STYLES.get(name, {})
            ax.plot(time, result["pred"][index], **style, label=DISPLAY_NAMES.get(name, name))
        ax.set(title=f"{label}: {ids[index]}", xlabel="time [s]")
        ax.grid(alpha=.25)
        if len(results) <= 3:
            ax.legend(ncol=1, fontsize=10, loc="lower left")
    if len(results) > 3:
        axes[0, 0].legend(ncol=1, fontsize=10)
    axes[1, 1].boxplot([value["corr"] for value in results.values()],
                       tick_labels=[DISPLAY_NAMES.get(name, name) for name in results], showfliers=False)
    axes[1, 1].set_title("episode waveform correlation")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return picks[1]


def plot_grid(rows, episode_id, fs, path, title):
    columns = max(len(panels) for _, panels in rows)
    time = np.arange(len(rows[0][1][0][1])) / fs
    fig, axes = plt.subplots(len(rows), columns, figsize=(3.1 * columns, 2.5 * len(rows)), squeeze=False)
    for r, (name, panels) in enumerate(rows):
        for c, (label, value, recorded) in enumerate(panels):
            ax = axes[r, c]
            ax.plot(time, value, color="#2a78d6", lw=.9, label="estimate")
            if recorded is None:
                label += " (estimate only)"
            else:
                ax.plot(time, recorded, color="black", lw=.9, alpha=.8, label="recorded")
                ax.legend(fontsize=10)
            ax.set_title(label, fontsize=12)
            ax.grid(alpha=.25)
            if c == 0:
                ax.set_ylabel(name)
            if r == len(rows) - 1:
                ax.set_xlabel("time [s]")
        for c in range(len(panels), columns):
            axes[r, c].axis("off")
    fig.suptitle(f"{title}: {episode_id}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric_grid(metric_sets, path):
    names = list(metric_sets)
    keys = list(next(iter(metric_sets.values())))
    fig, axes = plt.subplots(2, 3, figsize=(4.8 * max(1, len(names) / 4) * 3, 9))
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    for ax, key in zip(axes.flat, keys):
        ax.boxplot([metric_sets[name][key] for name in names],
                   tick_labels=[DISPLAY_NAMES.get(name, name) for name in names], showfliers=False)
        if key == "amplitude ratio":
            ax.axhline(1, color="#888888", lw=.8, linestyle="--")
        if key == "error-amplitude corr":
            ax.axhline(0, color="#888888", lw=.8, linestyle="--")
        ax.set_title(key, fontsize=12)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_spatial(speed, arrays, results, ids, fs, path):
    valid = np.isfinite(arrays["qc2_iri_m_per_km"]).any(1)
    candidates = np.flatnonzero(valid)
    target = np.median(results["rw"]["corr"])
    index = candidates[np.argmin(np.abs(results["rw"]["corr"][candidates] - target))]
    keep, time = np.isfinite(arrays["distance_m"][index]), np.arange(speed.shape[1]) / fs
    distance = arrays["distance_m"][index, keep]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(time, speed[index], color="#555555")
    for key, label in (("qc2_road_m", "qc2"), ("hc8_left_road_m", "hc8 left"), ("hc8_right_road_m", "hc8 right")):
        axes[1].plot(distance, arrays[key][index, keep], label=label)
    for key, label in (("qc2_iri_m_per_km", "qc2"), ("hc8_left_iri_m_per_km", "hc8 left"),
                       ("hc8_right_iri_m_per_km", "hc8 right")):
        axes[2].plot(distance, arrays[key][index, keep], label=label)
    axes[0].set_ylabel("speed [km/h]")
    axes[1].set_ylabel("road [m]")
    axes[2].set(ylabel="40 m IRI [m/km]", xlabel="distance [m]")
    for ax in axes:
        ax.grid(alpha=.25)
    axes[1].legend(ncol=3)
    axes[2].legend(ncol=3)
    fig.suptitle(f"Spatial road and IRI estimates only: {ids[index]}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hybrid_detail(target, baseline, hybrid, correction, innovation, ids, corr, fs, kf_name, path):
    order, time = np.argsort(corr), np.arange(target.shape[1]) / fs
    index = order[len(order) // 2]
    hybrid_name = f"{kf_name}_lstm"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(time, target[index], color="black", lw=1, label="recorded")
    axes[0].plot(time, baseline[index], color="#0000FF", linestyle="--", lw=1.1,
                 label=DISPLAY_NAMES[kf_name])
    axes[0].plot(time, hybrid[index], color="#FF0000", lw=1.1, label=DISPLAY_NAMES[hybrid_name])
    axes[1].plot(time, target[index] - baseline[index], color="black", lw=.9, label="target residual")
    axes[1].plot(time, correction[index], color="#FF0000", lw=.9, label="LSTM correction")
    axes[2].plot(time, innovation[index], color="#e67e22", lw=.9)
    axes[0].set_title("Bounce")
    axes[1].set_title("residual correction")
    axes[2].set_title("KF innovation")
    for ax in axes:
        ax.set_xlabel("time [s]")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=11)
    axes[1].legend(fontsize=11)
    fig.suptitle(f"Real-data 1-DOF {DISPLAY_NAMES[kf_name]} hybrid: {ids[index]}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_lstm_kf_detail(target, results, kinematic_state, fused_state, acceleration, ids, fs, path):
    order = np.argsort(results["lstm_kf"]["corr"])
    index, time_axis = order[len(order) // 2], np.arange(target.shape[1]) / fs
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(time_axis, target[index], color="black", lw=1.1, label="recorded")
    for name in ("kinematic_kf", "imu_lstm", "lstm_kf"):
        axes[0, 0].plot(time_axis, results[name]["pred"][index], **LINE_STYLES[name],
                        label=DISPLAY_NAMES[name])
    axes[0, 0].set_title("Bounce rate")
    axes[0, 0].legend(fontsize=11)
    for name in ("kinematic_kf", "imu_lstm", "lstm_kf"):
        axes[0, 1].plot(time_axis, target[index] - results[name]["pred"][index], **LINE_STYLES[name],
                        label=DISPLAY_NAMES[name])
    axes[0, 1].set_title("Estimation error")
    axes[1, 0].plot(time_axis, highpass(kinematic_state[index:index + 1, :, 0], .1, fs)[0],
                    **LINE_STYLES["kinematic_kf"], label="Kinematic KF z")
    axes[1, 0].plot(time_axis, highpass(fused_state[index:index + 1, :, 0], .1, fs)[0],
                    **LINE_STYLES["lstm_kf"], label="LSTM-KF z")
    axes[1, 0].set_title("High-pass displacement estimate only")
    axes[1, 0].legend(fontsize=11)
    axes[1, 1].plot(time_axis, acceleration[index], color="black", lw=.9, label="measured z_ddot")
    axes[1, 1].plot(time_axis, kinematic_state[index, :, 2], **LINE_STYLES["kinematic_kf"],
                    label="Kinematic KF z_ddot")
    axes[1, 1].plot(time_axis, fused_state[index, :, 2], **LINE_STYLES["lstm_kf"],
                    label="LSTM-KF z_ddot")
    axes[1, 1].set_title("Acceleration observation")
    axes[1, 1].legend(fontsize=11)
    for ax in axes.flat:
        ax.set_xlabel("time [s]")
        ax.grid(alpha=.25)
    fig.suptitle(f"Simplified paper-style LSTM-KF: {ids[index]}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_model_free(target, results, ids, fs, path):
    groups = tuple((kind.title(), [name for name in MODEL_FREE_PLOT_ORDER[kind] if name in results])
                   for kind in ("online", "offline"))
    time_axis = np.arange(target.shape[1]) / fs
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for column, (title, names) in enumerate(groups):
        reference = results[names[0]]["corr"]
        order = np.argsort(reference)
        index = order[len(order) // 2]
        recorded, = axes[0, column].plot(time_axis, target[index], color="black", lw=1.1,
                                         label="recorded", zorder=1)
        handles = {}
        draw_names = sorted(names, key=lambda value: MODEL_FREE_LINE_STYLES[value]["zorder"])
        for name in draw_names:
            handles[name], = axes[0, column].plot(
                time_axis, results[name]["pred"][index], **MODEL_FREE_LINE_STYLES[name],
                label=MODEL_FREE_DISPLAY_NAMES[name])
        axes[0, column].set(title=f"{title} median: {ids[index]}", xlabel="time [s]")
        axes[0, column].grid(alpha=.25)
        axes[0, column].legend(handles=[recorded] + [handles[name] for name in draw_names], fontsize=11)
        axes[1, column].boxplot([results[name]["corr"] for name in draw_names],
                                tick_labels=[MODEL_FREE_DISPLAY_NAMES[name] for name in draw_names],
                                showfliers=False)
        axes[1, column].tick_params(axis="x", rotation=20)
        axes[1, column].set_title(f"{title} correlation")
        axes[1, column].grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
