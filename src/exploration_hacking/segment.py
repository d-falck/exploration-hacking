from datasets import Dataset


class DataSegmenter:
    def __init__(self, dataset: Dataset, syst):
        self.dataset = dataset

    def merge_datasets(self, benign_train: Dataset, benign_test: Dataset, malign_train: Dataset, malign_test: Dataset) -> Dataset:
        ...