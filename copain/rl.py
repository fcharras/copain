import inspect
import threading
from queue import SimpleQueue
from dataclasses import dataclass, astuple

import numpy as np
import torch
import torch.cuda.amp as amp
from torch.utils.data.dataloader import default_collate

from skorch import NeuralNet
from skorch.utils import to_device


_default_nn_params = {
    name: param.default
    for name, param in inspect.signature(NeuralNet).parameters.items()
    if param.default is not inspect._empty
}


class _Pending:
    pass


@dataclass
class _Experience:
    state0: np.array = _Pending
    state1: np.array = _Pending
    action: np.array = _Pending
    reward: np.float32 = _Pending


class _ExperienceMemory:
    def __init__(self, size):
        self.size = size
        self._memory = np.empty(size, dtype=object)
        self._memory_writing_pointer = 0
        self._max_memory_idx = -1
        self._random = np.random.RandomState(42)

    def memorize_new_experience(self, experience):
        self._memory[self._memory_writing_pointer] = experience
        self._memory_writing_pointer = (self._memory_writing_pointer + 1) % self.size
        self._max_memory_idx = max(self._max_memory_idx, self._memory_writing_pointer)

    def get_random_experiences(self, nb_experiences):
        return self._random.choice(
            self._memory[: self._max_memory_idx], nb_experiences, replace=True
        )

    @property
    def nb_memories(self):
        return self._max_memory_idx + 1


class CopainAI(NeuralNet):
    def __init__(
        self,
        *,
        module,
        criterion,
        reward_fn,
        optimizer=_default_nn_params["optimizer"],
        lr=_default_nn_params["lr"],
        discount=0.9,
        exploration_rate=1,
        epoch_duration=600,  # TODO: implement
        max_epochs=_default_nn_params["max_epochs"],
        experience_replay_size=10000,
        experience_replay_burn_in=10000,
        training_batch_size=_default_nn_params["batch_size"],
        evaluation_batch_size=_default_nn_params["batch_size"],
        collate_fn=default_collate,
        random_action_fn=None,
        callbacks=_default_nn_params["callbacks"],
        warm_start=_default_nn_params["warm_start"],
        verbose=_default_nn_params["verbose"],
        device=_default_nn_params["device"],
        enable_amp=False,
        amp_init_scaling=20,
        **kwargs,
    ):
        batch_size = None
        dataset = None
        iterator_train = None
        iterator_valid = None
        train_split = None
        predict_nonlinearity = None
        super().__init__(
            module,
            criterion,
            optimizer,
            lr,
            max_epochs,
            batch_size,
            iterator_train,
            iterator_valid,
            dataset,
            train_split,
            callbacks,
            predict_nonlinearity,
            warm_start,
            verbose,
            device,
            **kwargs,
        )
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.reward_fn = reward_fn

        self.experience_replay_size = experience_replay_size
        self.experience_replay_burn_in = experience_replay_burn_in

        self.training_batch_size = training_batch_size
        self.evaluation_batch_size = evaluation_batch_size

        self.epoch_duration = epoch_duration
        self.collate_fn = collate_fn
        self.random_action_fn = random_action_fn
        self.enable_amp = enable_amp
        self.amp_init_scaling = amp_init_scaling

        self._connection_queues = dict()
        self._current_step = dict()
        self._requests_queue = SimpleQueue()

    def ask_action(self, X, run_id):
        self._requests_queue.put((run_id, False, X))
        return self._connection_queues[run_id].get()

    def tell_done(self, X, run_id):
        self._requests_queue.put((run_id, True, X))

    def register_run(self, run_id):
        if run_id in self._connection_queues:
            raise ValueError(f"A run with run_id {run_id} has already been registered.")

        self._connection_queues[run_id] = SimpleQueue()
        self._current_step[run_id] = _Experience()

    def unregister_run(self, run_id):
        if run_id not in self._connection_queues:
            raise ValueError(f"No run with run_id {run_id} in the run registry.")

        del self._connection_queues[run_id]
        del self._current_step[run_id]

    def serve_forever(self):
        self.is_serving = True
        __is_shut_down = self.__is_shut_down = threading.Event()
        try:
            self._serve_forever()
        finally:
            del self._requests_queue
            del self.__is_shut_down
            __is_shut_down.set()

    def _serve_forever(self):
        requests_queue = self._requests_queue
        random = np.random.RandomState(41)
        if self.training and not (self.warm_start and self.initialized_):
            self.initialize()
            amp_scaler = amp.GradScaler(
                init_scale=2.0 ** self.amp_init_scaling, enabled=self.enable_amp
            )
        experience = _ExperienceMemory(self.experience_replay_size)

        while self.is_serving:
            self._step(requests_queue, experience, amp_scaler, random)

    def _step(self, requests_queue, experience, amp_scaler, random):
        prediction_run_id = []
        evaluation_states = []
        while self._wait_for_next_request(
            requests_queue,
            len(prediction_run_id),
            burn_in=(experience.nb_memories >= self.experience_replay_burn_in),
        ):
            if prediction_request := self._receive_next_request(
                requests_queue, experience, random
            ):
                run_id, data = prediction_request
                prediction_run_id.append(run_id)
                evaluation_states.append(data)

        if self.training and (experience.nb_memories >= self.experience_replay_burn_in):
            (
                training_states,
                training_batch1,
                training_actions,
                rewards,
            ) = self._load_next_training_batch(experience)
            evaluation_states.extend(training_batch1)
            training_actions = to_device(default_collate(training_actions), self.device)
        with amp.autocast(enabled=self.enable_amp), torch.no_grad():
            scores_eval = self._evaluation_step(
                evaluation_states, prediction_run_id, training_actions
            )

        if not self.training:
            return

        self._train_step(
            training_states, training_actions, rewards, scores_eval, amp_scaler
        )

        # TODO: when cb is done this should be on batch cb
        if self.module_._embedding_bag.update_embeddings():
            del self.optimizer_
            self._initialize_optimizer()

    def _wait_for_next_request(self, requests_queue, nb_requests_preprocessed, burn_in):
        return (
            not requests_queue.empty()
            or (
                (not self.training or not burn_in)  #
                and (nb_requests_preprocessed == 0)
            )
        ) and nb_requests_preprocessed < self.evaluation_batch_size

    def _receive_next_request(self, requests_queue, experience, random):
        (
            run_id,
            done,
            data,
        ) = requests_queue.get()
        if not self.training:
            return (run_id, data)
        current_step = self._current_step[run_id]
        if current_step.state0 is not _Pending:
            current_step.state1 = data
            current_step.reward = self.reward_fn(
                current_step.state0,
                current_step.state1,
                current_step.action,
                done,
            )
            experience.memorize_new_experience(current_step)
        self._current_step[run_id] = _Experience()
        if done:
            return False

        self._current_step[run_id].state0 = data
        if experience.nb_memories < self.experience_replay_burn_in:
            random_action = not self.warm_start

        else:
            random_action = random.random_sample() < self.exploration_rate
        if not random_action:
            return (run_id, data)
        requested_action = self._random_action()
        self._current_step[run_id].action = requested_action
        self._connection_queues[run_id].put(requested_action)
        return False

    def _random_action(self):
        if self.random_action_fn is None:
            return torch.randint(self.module_.n_actions, size=(1,))[0]

        return self.random_action_fn()

    def _load_next_training_batch(self, experience):
        training_samples = experience.get_random_experiences(self.training_batch_size)
        training_samples = [
            astuple(training_sample) for training_sample in training_samples
        ]
        return map(list, zip(*training_samples))

    def _evaluation_step(self, evaluation_states, prediction_run_id, training_actions):
        nb_predictions_scheduled = len(prediction_run_id)

        evaluation_states = to_device(self.collate_fn(evaluation_states), self.device)
        self._set_training(training=False)
        scores_eval = self.module_(evaluation_states)

        if nb_predictions_scheduled == 0:
            return scores_eval[:, training_actions]

        for request_run_id, prediction in zip(
            prediction_run_id,
            scores_eval[:nb_predictions_scheduled].argmax(1).to("cpu"),
        ):
            self._current_step[request_run_id].action = prediction
            self._connection_queues[request_run_id].put(prediction)

        return scores_eval[nb_predictions_scheduled:, training_actions]

    def _train_step(
        self, training_states, training_actions, rewards, scores_eval, amp_scaler
    ):
        self.optimizer_.zero_grad(set_to_none=True)
        with amp.autocast(enabled=self.enable_amp):
            training_states = to_device(self.collate_fn(training_states), self.device)
            self._set_training(training=True)
            scores_training = self.module_(training_states)[:, training_actions]
            rewards = to_device(default_collate(rewards), self.device)
            expected_scores_training = rewards + self.discount * scores_eval
            loss = self.criterion_(scores_training, expected_scores_training)
        amp_scaler.scale(loss).backward()
        amp_scaler.step(self.optimizer_)
        amp_scaler.update()

    def set_training(self, training):
        if training is getattr(self, "training", None):
            raise ValueError

        self.training = training

    def shutdown(self):
        self.is_serving = False
        if hasattr(self, "_CopainAI__is_shut_down"):
            self.__is_shut_down.wait()
