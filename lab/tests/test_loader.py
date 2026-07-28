import sys
import time
import json
from dataclasses import asdict

sys.stdout.reconfigure(encoding="utf-8")

from loader import Dataset, View, GROUPS, resolve_features


class T:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *a):
        print(f"  [{time.perf_counter() - self.t:7.3f}s] {self.msg}")


def section(name):
    print(f"\n{'='*60}\n{name}\n{'='*60}")


# ---------------------------------------------------------------- registry
section("1. CHANNEL REGISTRY")
print("groups:", {g: len(c) for g, c in GROUPS.items()})
print("resolve 'imu' ->", resolve_features("imu"))
print("resolve mix   ->", resolve_features(["imu", "Pitch_rate_6D"]))

# ---------------------------------------------------------------- dataset
section("2. DATASET BUILD")
with T("Dataset() 1st (cold scan)"):
    ds = Dataset()
with T("Dataset() 2nd (memo)"):
    Dataset()
print(ds)
print("names:", ds.names)
ds.summary()

# ---------------------------------------------------------------- navigation
section("3. NAVIGATION: driver -> episode -> signal")
drv = ds[ds.names[0]]
print("driver :", drv)
ep = drv[0]
print("episode:", ep)
print("event  :", ep.event_time, "| vehicle:", ep.vehicle,
      "| video:", ep.video.name if ep.video else None)
with T("ep.signals 1st (read_csv)"):
    sig = ep.signals
with T("ep.signals 2nd (cached)"):
    sig = ep.signals
ch = sig["Pitch_rate_6D"]
print("signal :", ch, "|", ch.axis_label)
print("group  :", sig.imu, "->", sig.imu.displays)
ep.info()

# ---------------------------------------------------------------- flat queries
section("4. FLAT QUERIES")
print("label=True   :", ds.episodes.filter(label=True))
print("driver+label :", ds.episodes.filter(driver=ds.names[0], label=False))
print("drivers=[a,b]:", ds.episodes.filter(driver=ds.names[:2]))
print("by_driver    :", {k: len(v) for k, v in ds.episodes.by_driver().items()})

# ---------------------------------------------------------------- view
section("5. VIEW -> TENSOR")
view = View(features="imu", around=(-1, 2))
with T("view(driver) 1st [read_csv]"):
    X, y = view(drv)
print("   ->", X.shape, "pos", int(y.sum()))
with T("view(driver) 2nd [cached signals]"):
    X, y = view(drv)
with T("view(episode) single"):
    m = view(ep)
print("   ->", m.shape)

view_s = View(features=["imu", "derived"], around=(-2, 2), downsample=2, smooth=(10.0, 2))
with T("view smooth+downsample"):
    Xs, ys = view_s(drv)
print("   ->", Xs.shape)

with T("view(ds.episodes) ALL drivers"):
    Xa, ya = view(ds.episodes)
print("   ->", Xa.shape, "pos", int(ya.sum()))

FEATURES = ["Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal", "IMU_LatAccelVal",
            "IMU_YawRtVal", "Roll_rate_6D", "SAS_AnglVal", "SAS_SpdVal", "IMU_RollRtVal",
            "VCU_AccPedDepVal", "IEB_StrkDpthPcVal", "IEB_BrkActvSta", "IEB_EstTtlBrkFrcNmV"]
with T("view 13 features, ALL drivers"):
    Xf, yf = View(features=FEATURES, around=(-1, 2))(ds.episodes)
print("   ->", Xf.shape)

# ---------------------------------------------------------------- view spec
section("6. VIEW SPEC (identity / json)")
print("view:", view_s)
spec = json.dumps(asdict(view_s))
print("json:", spec)
back = View(**json.loads(spec))
print("roundtrip == original:", back == view_s)
print("hash equal           :", hash(back) == hash(view_s))
print("train/test same view :", View(features="imu", around=(-1, 2)) == view)

print("\nDONE.")
