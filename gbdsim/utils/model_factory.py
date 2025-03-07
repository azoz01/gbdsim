from gbdsim.baselines.dataset2vec import Dataset2VecWrapped
from gbdsim.experiment_config import Dataset2VecConfig, GBDSimConfig
from gbdsim.model.gbdsim import GBDSim
from gbdsim.utils.protocols import DatasetDistanceCalculator


class ModelFactory:

    @staticmethod
    def get_model(
        config: Dataset2VecConfig | GBDSimConfig,
    ) -> DatasetDistanceCalculator:
        if isinstance(config, Dataset2VecConfig):
            return Dataset2VecWrapped()
        else:
            return GBDSim.from_config(config)
