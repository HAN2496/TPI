import sys

sys.modules["models"] = sys.modules[__name__]  # 구 model.joblib pickle의 models.* 경로 호환
