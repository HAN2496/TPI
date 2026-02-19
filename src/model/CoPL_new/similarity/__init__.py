from src.model.CoPL_new.similarity.base import ItemSimilarityBuilder
from src.model.CoPL_new.similarity.pca import PCASimilarity, KernelPCASimilarity
from src.model.CoPL_new.similarity.vae import VAESimilarity
from src.model.CoPL_new.similarity.dtw import DTWSimilarity

SIMILARITY_REGISTRY = {
    "pca": PCASimilarity,
    "kernel_pca": KernelPCASimilarity,
    "vae": VAESimilarity,
    "dtw": DTWSimilarity,
}


def build_similarity(method: str = "pca") -> ItemSimilarityBuilder:
    """similarity_method 문자열로 적절한 빌더 인스턴스를 생성합니다."""
    cls = SIMILARITY_REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown similarity method '{method}'. "
            f"Available: {list(SIMILARITY_REGISTRY.keys())}")
    return cls()
