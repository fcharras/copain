import inspect
from dataclasses import dataclass

import numpy as np

from skorch import NeuralNet

from threading import Lock


_default_nn_params = {
    name: param.default
    for name, param in inspect.signature(NeuralNet).parameters.items()
    if param.default is not inspect._empty
}


class PendingData:
    pass


@dataclass
class Experience:
    state0: np.array = PendingData
    state1: np.array = PendingData
    action: np.array = PendingData
    done: bool = PendingData
    reward: np.float32 = PendingData
    score0: int = PendingData
    score1: int = PendingData


class ExperienceMemory:
    def __init__(self, size):
        self.size = size
        self._memory = np.empty(size, dtype=object)
        self._memory_writing_pointer = 0
        self._max_memory_idx = -1
        self._random = np.random.default_rng(42)
        self._lock = Lock()

    def memorize_new_experience(self, experience):
        with self._lock:
            self._memory[self._memory_writing_pointer] = experience
            self._max_memory_idx = max(self._max_memory_idx, self._memory_writing_pointer)
            self._memory_writing_pointer = (self._memory_writing_pointer + 1) % self.size

    def get_random_experiences(self, nb_experiences):
        with self._lock:
            return self._random.choice(
                self._memory[: self._max_memory_idx], nb_experiences, replace=True
            )

    def apply(self, fn):
        for i in range(self.nb_memories):
            self._memory[i] = fn(self._memory[i])

    @property
    def nb_memories(self):
        return self._max_memory_idx + 1
