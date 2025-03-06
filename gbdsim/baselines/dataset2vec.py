import torch.nn.functional as F
from dataset2vec.model import Dataset2Vec
from torch import Tensor


class Dataset2VecWrapped(Dataset2Vec):

    def calculate_dataset_distance(
        self, X1: Tensor, y1: Tensor, X2: Tensor, y2: Tensor
    ) -> Tensor:
        enc1 = self(X1, y1).unsqueeze(0)
        enc2 = self(X2, y2).unsqueeze(0)
        return 1 - F.cosine_similarity(enc1, enc2)
