# name: (group, display, latex, unit)   latex=None -> status/non-physical
CHANNELS = {
    "IMU_RollRtVal":       ("imu", "RollRate", r"\dot{\phi}", "deg/s"),
    "IMU_VerAccelVal":     ("imu", "VertAccel", r"a_z", "m/s^2"),
    "IMU_YawRtVal":        ("imu", "YawRate", r"\dot{\psi}", "deg/s"),
    "IMU_LatAccelVal":     ("imu", "LatAccel", r"a_y", "m/s^2"),
    "IMU_LongAccelVal":    ("imu", "LongAccel", r"a_x", "m/s^2"),

    "WHL_SpdRRVal":        ("wheel", "WheelSpd RR", r"v_{RR}", "km/h"),
    "WHL_SpdFLVal":        ("wheel", "WheelSpd FL", r"v_{FL}", "km/h"),
    "WHL_SpdFRVal":        ("wheel", "WheelSpd FR", r"v_{FR}", "km/h"),
    "WHL_SpdRLVal":        ("wheel", "WheelSpd RL", r"v_{RL}", "km/h"),

    "SAS_AnglVal":         ("steering", "SteerAngle", r"\delta", "deg"),
    "SAS_SpdVal":          ("steering", "SteerRate", r"\dot{\delta}", "deg/s"),
    "SAS_IntSta":          ("steering", "SteerInit", None, ""),
    "SAS_Crc1Val":         ("steering", "SteerCRC", None, ""),
    "SAS_AlvCnt1Val":      ("steering", "SteerAliveCnt", None, ""),

    "MCU_Mg1EstTqVal":     ("motor", "MotorTq1", r"T_{m1}", "Nm"),
    "MCU_Mg1EstTqPcVal":   ("motor", "MotorTq1", r"T_{m1,\%}", "%"),
    "MCU_Mg2EstTqVal":     ("motor", "MotorTq2", r"T_{m2}", "Nm"),
    "MCU_Mg2EstTqPcVal":   ("motor", "MotorTq2", r"T_{m2,\%}", "%"),

    "VCU_MotTqCmdRearVal": ("vcu", "TqCmd Rear", r"T_{cmd,r}", "Nm"),
    "VCU_MotTqCmdFrntVal": ("vcu", "TqCmd Front", r"T_{cmd,f}", "Nm"),
    "VCU_GearPosSta":      ("vcu", "Gear", None, ""),
    "VCU_AccPedDepVal":    ("vcu", "AccelPedal", r"\alpha_{ped}", "%"),

    "IEB_StrkDpthPcVal":   ("brake", "BrakeStroke", r"s_{brk}", "%"),
    "IEB_BrkActvSta":      ("brake", "BrakeActive", None, ""),
    "IEB_EstTtlBrkFrcNmV": ("brake", "BrakeForce", r"F_{brk}", "N"),
    "ABS_ActvSta":         ("brake", "ABSActive", None, ""),

    "Bounce_rate_6D":      ("derived", "BounceRate", r"\dot{z}", "m/s"),
    "Roll_rate_6D":        ("derived", "RollRate6D", r"\dot{\phi}_{6D}", "deg/s"),
    "Pitch_rate_6D":       ("derived", "PitchRate", r"\dot{\theta}", "deg/s"),

    "FCA_OnOffEquipSta":   ("fca", "FCA On", None, ""),
    "FCA_WrngLvlSta":      ("fca", "FCA WarnLvl", None, ""),
    "FCA_WrngSndSta":      ("fca", "FCA WarnSnd", None, ""),
}

GROUPS = {}
for _n, (_g, *_rest) in CHANNELS.items():
    GROUPS.setdefault(_g, []).append(_n)

CHANNEL_GROUP = {n: v[0] for n, v in CHANNELS.items()}
ALL = list(CHANNELS)


def resolve_features(features):
    if features is None:
        return list(ALL)
    if isinstance(features, str):
        features = [features]
    out = []
    for f in features:
        out += GROUPS[f] if f in GROUPS else [f]
    return out
