from dataset2vec.config import Dataset2VecConfig as Dataset2VecConfigOriginal
from torch import nn
from typing_protocol_intersection import ProtocolIntersection as Has

from gbdsim.baselines.dataset2vec import Dataset2VecWrapped
from gbdsim.experiment_config import Dataset2VecConfig, GBDSimConfig
from gbdsim.model.gbdsim import GBDSim
from gbdsim.training.origin_classification import OriginClassificationLearner
from gbdsim.utils.protocols import DatasetDistanceCalculator


class ModelFactory:

    @staticmethod
    def get_model(
        config: Dataset2VecConfig | GBDSimConfig,
    ) -> Has[DatasetDistanceCalculator, OriginClassificationLearner]:
        if isinstance(config, Dataset2VecConfig):
            return Dataset2VecWrapped(
                Dataset2VecConfigOriginal(
                    activation_cls=nn.LeakyReLU,
                    f_dense_hidden_size=256,
                    f_block_repetitions=1,
                    f_out_size=256,
                    g_layers_sizes=[256, 512, 256],
                    h_dense_hidden_size=256,
                    h_block_repetitions=1,
                    output_size=128,
                )
            )
        else:
            return GBDSim.from_config(config)
