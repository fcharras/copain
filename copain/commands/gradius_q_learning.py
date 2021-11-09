FCEUX_EXECUTABLE = "/usr/bin/fceux"

ROM_PATH = "./Gradius (USA).zip"
ROM_HASH = "49c0ffbd6fca692e7c567b4d72ae0847"

DISPLAY_FCEUX_GUI = True
ENABLE_GAME_GENIE = False
ENABLE_SOUND = False

SPEEDMODE = "maximum"
RENDER_SPRITES = False
RENDER_BACKGROUND = False

TMP_FOLDER = "/tmp/"

NUM_RUNNERS = 3
THREADED_SOCKET = True
THREADED_REQUESTS = True

FRAME_PER_ACTION = 30

DEVICE = "cuda"
EPOCH_DURATION = 600

EXPERIENCE_REPLAY_SIZE = 100000
EXPERIENCE_REPLAY_BURN_IN = 30000

ENABLE_TRAINING = True

ENABLE_AMP = True
AMP_INIT_SCALING = 20
TRAINING_BATCH_SIZE = 64
STARTING_NB_EMBEDDINGS = 100000
NB_EMBEDDINGS_STEP = 50000
LR = 0.001

EVALUATION_BATCH_SIZE = 32

NN_DEPTH = 1
EMBEDDING_SIZE = 256
HIDDEN_DIM = 8192
P_DROPOUT = 0.5

DISCOUNT_RATE = 0.9
EXPLORATION_RATE = 0.05

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from copain import CopainRun
from copain.copain_driver import P1, DIRECTIONS
from copain.rl import CopainAI
from copain.nn import CopainANN


_GRADIUS_MEMORY_SIZE = 0x8000
_GRADIUS_SLOT_SIZE = 0x100
_GRADIUS_N_ACTIONS = 9 * 2 * 3  # 54


def gradius_random_action():
    return torch.randint(36, size=(1,))[0]


class GradiusLoopFn:
    MEMORY_BYTERANGE_START = np.uint16(0).tobytes()
    MEMORY_BYTERANGE_LENGTH = np.uint16(_GRADIUS_MEMORY_SIZE)
    MEMORY_BYTERANGE_BLENGTH = MEMORY_BYTERANGE_LENGTH.tobytes()

    IS_ALIVE_ADDRESS = 0x100

    SKIP_MULTIPLIERS = [1, 2, 4]

    def __init__(self, frame_per_action):
        self._skip_after_input = frame_per_action - 1

    def __call__(self, handler, copain_ai, run_metadata):
        self.handler = handler
        self._fire_frame = True

        runner_id = run_metadata.runner_id
        copain_ai.register_run(runner_id)

        while True:
            handler.emu_poweron()
            self._starting_sequence()
            while True:
                gamestate = self._get_gamestate()
                action = copain_ai.ask_action(gamestate, runner_id)
                time, inputs = self._get_inputs(action)
                self._frameadvance_with_inputs_and_autofire(inputs)
                inputs_directions = [press for press in inputs if press in DIRECTIONS]
                for i in range(self.SKIP_MULTIPLIERS[time] * self._skip_after_input):
                    gamestate = self._get_gamestate()
                    must_reset = self._done_condition(gamestate)
                    if must_reset:
                        copain_ai.tell_done(gamestate, runner_id)
                        break
                    self._frameadvance_with_inputs_and_autofire(inputs_directions)
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


def gradius_loop_fn_init():
    return GradiusLoopFn(FRAME_PER_ACTION)


def _gradius_parse_action_nb(action):
    time = torch.div(action, 18, rounding_mode="floor")
    action = action % 18
    press_b = torch.div(action, 9, rounding_mode="floor")
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
        + 8 * score_diff
        - 64 * int(done)
    ) / 83


if __name__ == "__main__":

    copain_ai = CopainAI(
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
        criterion=torch.nn.HuberLoss,
        reward_fn=_gradius_reward_fn,
        optimizer=torch.optim.Adam,
        optimizer__amsgrad=True,
        lr=LR,
        discount=DISCOUNT_RATE,
        exploration_rate=EXPLORATION_RATE,
        epoch_duration=EPOCH_DURATION,
        max_epochs=100,
        experience_replay_size=EXPERIENCE_REPLAY_SIZE,
        experience_replay_burn_in=EXPERIENCE_REPLAY_BURN_IN,
        training_batch_size=TRAINING_BATCH_SIZE,
        evaluation_batch_size=EVALUATION_BATCH_SIZE,
        collate_fn=default_collate,
        random_action_fn=gradius_random_action,
        enable_amp=ENABLE_AMP,
        amp_init_scaling=AMP_INIT_SCALING,
        device=DEVICE,
    )
    copain_ai.set_training(ENABLE_TRAINING)

    copain = CopainRun(
        rom_path=ROM_PATH,
        rom_hash=ROM_HASH,
        loop_fn_init=gradius_loop_fn_init,
        threaded_socket=THREADED_SOCKET,
        threaded_requests=THREADED_REQUESTS,
        num_runners=NUM_RUNNERS,
        enable_game_genie=ENABLE_GAME_GENIE,
        display_fceux_gui=DISPLAY_FCEUX_GUI,
        visible_enable_sound=ENABLE_SOUND,
        visible_speedmode=SPEEDMODE,
        visible_render_sprites=RENDER_SPRITES,
        visible_render_background=RENDER_BACKGROUND,
        tmp_folder=TMP_FOLDER,
        fceux_executable=FCEUX_EXECUTABLE,
        copain_ai=copain_ai,
        threaded_ai=True,
    ).run()
