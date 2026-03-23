from .base import ItemSimilarityBuilder
from .pca import PCASimilarity, KernelPCASimilarity
from .vae import VAESimilarity
from .dtw import DTWSimilarity

SIMILARITY_REGISTRY = {
    "pca": PCASimilarity,
    "kernel_pca": KernelPCASimilarity,
    "vae": VAESimilarity,
    "dtw": DTWSimilarity,
}


def build_similarity(method: str = "pca") -> ItemSimilarityBuilder:
    cls = SIMILARITY_REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown similarity method '{method}'. "
            f"Available: {list(SIMILARITY_REGISTRY.keys())}")
    return cls()
