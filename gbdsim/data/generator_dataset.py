from typing import Any, Callable

from torch.utils.data import IterableDataset


class GeneratorDataset(IterableDataset):

    def __init__(
        self,
        batch_generation_fn: Callable[[], Any],
        length: int,
        repeatable_output: bool = False,
    ):
        self.length = length
        self.repeatable_output = repeatable_output
        if repeatable_output:
            self.generator = self.__get_repeatable_generation_fn(
                batch_generation_fn, length
            )
        else:
            self.generator = self.__get_non_repeatable_generation_fn(
                batch_generation_fn, length
            )

    def __get_non_repeatable_generation_fn(
        self, batch_generation_fn: Callable[[], Any], length: int
    ):
        def generator():
            for _ in range(length):
                yield batch_generation_fn()
            raise StopIteration()

        return generator

    def __get_repeatable_generation_fn(
        self, batch_generation_fn: Callable[[], Any], length: int
    ):
        self.examples = [batch_generation_fn() for _ in range(length)]

        def generator():
            for ex in self.examples:
                yield ex

        return generator

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return self.length
