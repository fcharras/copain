FCEUX_EXECUTABLE = "/usr/bin/fceux"

ROM_PATH = "./Gradius (USA).zip"
ROM_HASH = "49c0ffbd6fca692e7c567b4d72ae0847"

DISPLAY_FCEUX_GUI = True
ENABLE_GAME_GENIE = False
ENABLE_SOUND = False

SPEEDMODE = "maximum"
RENDER_SPRITES = True
RENDER_BACKGROUND = True

TMP_FOLDER = "/tmp/"

FRAME_PER_ACTION = 16

MAX_NB_ORDINARY_MEMORIES = 5000
MAX_NB_OUTSTANDING_MEMORIES = 10000
REWARD_CUTOFF = 0.5
MEMORY_BURN_IN = 1000

TORCH_NUM_THREADS = 1
DEVICE="cuda"
AMP_ENABLED = True
AMP_INIT_SCALE = 20
BATCH_SIZE = 64
STARTING_NB_EMBEDDINGS = 100000
NB_EMBEDDINGS_STEP = 50000
LR = 0.001

NN_DEPTH = 2
EMBEDDING_SIZE = 64
HIDDEN_DIM = 4096
P_DROPOUT = 0.5

DISCOUNT_RATE = 0.5
EXPLORATION_RATE = 0.001
RANDOM_START = True
PLAYER_WEIGHT_REFRESH_RATE = 60


import io
from dataclasses import astuple
from time import sleep, perf_counter
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from sklearn.base import clone

from skorch.callbacks import Callback

from torch.utils.data import IterableDataset

from copain import CopainRun
from copain.copain_driver import P1, DIRECTIONS
from copain.nn import NeuralNet, CopainANN

from copain.rl import Experience, ExperienceMemory

_GRADIUS_MEMORY_SIZE = 0x8000
_GRADIUS_SLOT_SIZE = 0x100
_GRADIUS_N_ACTIONS = 9 * 2 * 3  # 54


def gradius_random_action():
    return torch.randint(36, size=(1,))[0]


class GradiusIALoopFn:
    MEMORY_BYTERANGE_START = np.uint16(0).tobytes()
    MEMORY_BYTERANGE_LENGTH = np.uint16(_GRADIUS_MEMORY_SIZE)
    MEMORY_BYTERANGE_BLENGTH = MEMORY_BYTERANGE_LENGTH.tobytes()

    IS_ALIVE_ADDRESS = 0x100

    SKIP_MULTIPLIERS = [1, 2, 4]

    def __init__(self, exploration_rate, frame_per_action, player, trainer, weight_sync_lock,
                 experience_memory, weight_refresh_rate, random_start):
        self._skip_after_input = frame_per_action - 1
        self.player = player
        self.trainer = trainer
        self.weight_sync_lock = weight_sync_lock
        self.weight_refresh_rate = weight_refresh_rate
        self.experience_memory = experience_memory
        self.exploration_rate = exploration_rate
        self._random = np.random.default_rng(42)
        self.random_start = random_start

    def __call__(self, handler):
        self.handler = handler
        self._fire_frame = True

        timer = perf_counter()

        while True:
            handler.emu_poweron()
            self._starting_sequence()
            experience = None
            while True:
                nb_frameadvances = np.inf
                i = 0

                while (i < nb_frameadvances):
                    gamestate = self._get_gamestate()
                    must_reset = self._done_condition(gamestate)
                    if must_reset:
                        if experience is not None:
                            experience.state1 = gamestate
                            experience.done = True
                            experience.score1 = 0
                            self.experience_memory.memorize_new_experience(experience)
                        break

                    if i == 0:
                        if (perf_counter() - timer) >= self.weight_refresh_rate:
                            print("syncing...")
                            del self.player.module_
                            del self.player.initialized_
                            with self.weight_sync_lock, io.BytesIO() as ram_buffer:
                                torch.save(self.trainer.module_, ram_buffer)
                                ram_buffer.seek(0)
                                self.player.module_ = torch.load(
                                    ram_buffer, map_location=torch.device("cpu"))
                                del ram_buffer
                                self.player.initialized_ = True
                                self.update_experience_scores()
                            print("synced successfully!")
                            timer = perf_counter()

                        scores = self.player.predict_proba(
                            gamestate.reshape((1, -1)))
                        if experience is None or self._random.random() < self.exploration_rate:
                            action = self._random.integers(scores.shape[1])
                        else:
                            action = scores.argmax(1)[0]
                        score = scores[0, action]
                        if experience is not None:
                            experience.state1 = gamestate
                            experience.score1 = score
                            experience.done = False
                            self.experience_memory.memorize_new_experience(experience)
                        experience = Experience(state0=gamestate,
                                                action=action,
                                                score0=score)

                        time, inputs = self._get_inputs(action)
                        inputs_directions = [press for press in inputs if press in DIRECTIONS]
                        nb_frameadvances = self.SKIP_MULTIPLIERS[time] * self._skip_after_input
                        self._frameadvance_with_inputs_and_autofire(inputs)
                    else:
                        self._frameadvance_with_inputs_and_autofire(inputs_directions)

                    i += 1

                if must_reset:
                    break

    def _starting_sequence(self):
        for i in range(2):
            self.handler.emu_nframeadvance(np.uint8(60).tobytes())
            self.handler.joypad_set(P1, "start")

        while True:
            if self._is_alive(self._get_gamestate()):
                break
            self.handler.emu_frameadvance()

    def _frameadvance_with_inputs_and_autofire(self, inputs):
        if self._fire_frame:
            self.handler.joypad_set(P1, "A", *inputs)
        else:
            self.handler.joypad_set(P1, *inputs)

        self.handler.emu_frameadvance()
        self._fire_frame = not self._fire_frame

    def _done_condition(self, gamestate):
        return not self._is_alive(gamestate)

    def _is_alive(self, gamestate):
        return gamestate[self.IS_ALIVE_ADDRESS] == 1

    def _get_gamestate(self):
        # TODO: include ppu ?
        return self.handler.memory_readbyterange(
            self.MEMORY_BYTERANGE_START,
            self.MEMORY_BYTERANGE_LENGTH,
            self.MEMORY_BYTERANGE_BLENGTH,
        ).copy()

    def _get_inputs(self, action):
        time, press_b, direction = _gradius_parse_action_nb(action)
        inputs = []
        if press_b == 1:
            inputs.append("B")
        if (direction < 2) or direction == 7:
            inputs.append("up")
        if (direction > 0) and (direction < 4):
            inputs.append("right")
        if (direction > 2) and (direction < 6):
            inputs.append("down")
        if (direction > 4) and (direction < 8):
            inputs.append("left")

        return time, inputs

    def update_experience_scores(self):
        self.trainer.module_.to(torch.device("cpu"))
        self.player.module_.to(torch.device("cuda"))
        self.player.device = "cuda"
        experiences = self.experience_memory.get_all_experiences()
        current_idx = 0
        batch_size = self.player.batch_size
        while current_idx < len(experiences):
            experience_batch = experiences[current_idx:(current_idx+batch_size)]
            gamestates = []
            actions = []
            for experience in experience_batch:
                experience.score1 = None
                gamestates.append(experience.state1)
                actions.append(experience.action)
            gamestates = np.vstack(gamestates)
            actions = np.array(actions)
            scores = self.player.predict_proba(gamestates)
            scores = scores[np.arange(scores.shape[0]), actions]
            for i, experience in enumerate(experience_batch):
                experience.score1 = scores[i]
            current_idx += batch_size
        self.player.module_.to(torch.device("cpu"))
        self.player.device = "cpu"
        self.trainer.module_.to(torch.device("cuda"))


def _gradius_parse_action_nb(action):
    time = action // 18
    action = action % 18
    press_b = action // 9
    direction = action % 9
    return time, press_b, direction


def _gradius_get_score(data):
    score_low = data[0x07E4]
    score_med = data[0x07E5]
    score_high = data[0x07E6]
    score = (
        (score_low % 16)
        + 10 * torch.div(score_low, 16, rounding_mode="floor")
        + 100 * (score_med % 16)
        + 1000 * torch.div(score_med, 16, rounding_mode="floor")
        + 10000 * (score_high % 16)
        + 100000 * torch.div(score_high, 16, rounding_mode="floor")
    )
    return score


def _gradius_reward_fn(data_before, data_after, action, done):
    score_diff = (_gradius_get_score(data_after) - _gradius_get_score(data_before)) / 10
    time, press_b, direction = _gradius_parse_action_nb(action)
    any_direction = direction < 8
    return (
        69
        - int(any_direction)
        + 2 * time
        - 4 * press_b
        + 32 * score_diff
        - 64 * int(done)
#    ) / 83
    ) / 100

class GradiusExperienceMemory:

    def __init__(self, ordinary_memory_size, oustanding_memory_size, reward_cutoff):
        self.ordinary_memory = ExperienceMemory(ordinary_memory_size)
        self.outstanding_memory = ExperienceMemory(oustanding_memory_size)
        self.reward_cutoff = reward_cutoff
        self._random = np.random.default_rng(42)

    def memorize_new_experience(self, experience):
        experience.reward = _gradius_reward_fn(
            experience.state0, experience.state1, experience.action, experience.done)

        memory = (self.outstanding_memory if (experience.reward < self.reward_cutoff)
                  else self.ordinary_memory)
        memory.memorize_new_experience(experience)

    def get_random_experiences(self, nb_experiences):
        is_outstanding = (self._random.random(nb_experiences) < (
            self.outstanding_memory.nb_memories/(self.outstanding_memory.nb_memories
                                                 + self.ordinary_memory.nb_memories)))
        nb_outstanding = is_outstanding.sum()
        outstanding_memories = self.outstanding_memory.get_random_experiences(nb_outstanding)
        ordinary_memories = self.ordinary_memory.get_random_experiences(
            nb_experiences-nb_outstanding)
        memories = np.hstack((outstanding_memories, ordinary_memories))
        self._random.shuffle(memories)
        return memories

    def get_all_experiences(self):
        return np.hstack((self.ordinary_memory.get_all_experiences(),
                          self.outstanding_memory.get_all_experiences()))

    @property
    def nb_memories(self):
        return self.ordinary_memory.nb_memories + self.outstanding_memory.nb_memories


class MemoryStreamDataset(IterableDataset):
    def __init__(self, experience_memory, min_nb_memories):
        self.experience_memory = experience_memory
        self.min_nb_memories = min_nb_memories

    def __iter__(self):
        while self.experience_memory.nb_memories < self.min_nb_memories:
            sleep(1)

        while True:
            experience = self.experience_memory.get_random_experiences(1)[0]
            (data_before, data_after, action, done, reward,
             score_before, score_after) = astuple(experience)
            yield data_before, (action, reward, done, score_after)


class RLLoss(torch.nn.Module):

    def __init__(self, base_criterion, discount):
        super().__init__()
        self.base_criterion = base_criterion()
        self.discount = discount

    def forward(self, prediction, y):
        action, reward, done, score_after = y
        prediction = prediction[torch.arange(prediction.size(0)), action]
        actual = reward + self.discount * score_after * (~done)
        return self.base_criterion(prediction, actual)


class UpdateEmbeddings(Callback):
    def on_batch_end(self, net, *args, **kwargs):
        if net.module_._embedding_bag.update_embeddings():
            del net.optimizer_
            net._initialize_optimizer()


class LockCompute(Callback):
    def __init__(self, lock):
        self.lock = lock
    def on_batch_begin(self, *args, **kwargs):
        self.lock.acquire()
    def on_batch_end(self, *args, **kwargs):
        self.lock.release()

if __name__ == "__main__":

    torch.set_num_threads(1)

    weight_sync_lock = Lock()

    player = NeuralNet(
        module=CopainANN,
        module__n_actions=_GRADIUS_N_ACTIONS,
        module__input_dim=_GRADIUS_MEMORY_SIZE,
        module__nb_values_per_dim=_GRADIUS_SLOT_SIZE,
        module__starting_nb_embeddings=STARTING_NB_EMBEDDINGS,
        module__nb_embeddings_step=NB_EMBEDDINGS_STEP,
        module__depth=NN_DEPTH,
        module__embedding_size=EMBEDDING_SIZE,
        module__hidden_dim=HIDDEN_DIM,
        module__p_dropout=P_DROPOUT,
        module__initialize_fn=torch.nn.init.kaiming_normal_,
        module__initialize_fn_kwargs=dict(nonlinearity="relu"),
        criterion=RLLoss,
        criterion__base_criterion = torch.nn.HuberLoss,
        criterion__discount = DISCOUNT_RATE,
        optimizer=torch.optim.Adam,
        optimizer__amsgrad=True,
        lr=LR,
        max_epochs=100,
        batch_size=BATCH_SIZE,
        amp_enabled=AMP_ENABLED,
        amp_init_scale=AMP_INIT_SCALE,
        device="cpu",
        train_split=None,
    )

    trainer = clone(player)
    trainer.set_params(device=DEVICE,
                       callbacks=[("Embedding_Mgt", UpdateEmbeddings()),
                                  ("sync_callback", LockCompute(weight_sync_lock))])

    experience_memory = GradiusExperienceMemory(MAX_NB_ORDINARY_MEMORIES,
                                                MAX_NB_OUTSTANDING_MEMORIES, REWARD_CUTOFF)

    player.initialize()

    # create experience memory using EXPERIENCE_REPLAY_SIZE (EXPERIENCE_REPLAY_BURN_IN ?)
    def gradius_loop_fn_init():
        return GradiusIALoopFn(EXPLORATION_RATE, FRAME_PER_ACTION, player, trainer,
                               weight_sync_lock, experience_memory, PLAYER_WEIGHT_REFRESH_RATE,
                               RANDOM_START)

    copain = CopainRun(
        rom_path=ROM_PATH,
        rom_hash=ROM_HASH,
        loop_fn_init=gradius_loop_fn_init,
        enable_game_genie=ENABLE_GAME_GENIE,
        display_fceux_gui=DISPLAY_FCEUX_GUI,
        enable_sound=ENABLE_SOUND,
        speedmode=SPEEDMODE,
        render_sprites=RENDER_SPRITES,
        render_background=RENDER_BACKGROUND,
        tmp_folder=TMP_FOLDER,
        fceux_executable=FCEUX_EXECUTABLE,
    )

    with ThreadPoolExecutor(max_workers=1) as thread_executor:
        thread_executor.submit(copain.run)
        dataset = MemoryStreamDataset(experience_memory, min_nb_memories=MEMORY_BURN_IN)
        trainer.fit(dataset)

# TODO:
# sync in separate thread, and update memory with gpu
# several player threads
# one of the players is normal speed to spectate
# reduce input space (no bonus, no time)
# prefill the memory with a pre-saved movie ?
# clean mess
