FCEUX_EXECUTABLE = "/usr/bin/fceux"

ROM_PATH = "./Gradius (USA).zip"
ROM_HASH = "49c0ffbd6fca692e7c567b4d72ae0847"

DISPLAY_FCEUX_GUI = True
ENABLE_GAME_GENIE = False
ENABLE_SOUND = False

SPEEDMODE = "normal"
RENDER_SPRITES = True
RENDER_BACKGROUND = True

TMP_FOLDER = "/tmp/"

NUM_RUNNERS = 1
THREADED_SOCKET = True
THREADED_REQUESTS = True

FRAME_PER_ACTION = 16
NUMBER_OF_SAVESTATES = 10
ACTIONS_PER_SAVESTATES = 12

from collections import deque
from weakref import WeakKeyDictionary

import numpy as np

from copain import CopainRun
from copain.copain_driver import P1


class GradiusLoopFn:
    ONE = np.uint16(1)
    bONE = ONE.tobytes()
    IS_ALIVE_ADDRESS = np.uint16(0x100).tobytes()

    X_AXIS_ADDRESS = np.uint16(0x360).tobytes()  # range 16 (left) - 240 (right)
    X_AXIS_BALANCE = np.uint16(80)
    # X_AXIS_BALANCE = np.uint16(146)
    Y_AXIS_ADDRESS = np.uint16(0x320).tobytes()  # range 16 (top) - 192 (right)
    # Y_AXIS_BALANCE = np.uint16(96)
    Y_AXIS_BALANCE = np.uint16(50)

    DIRECTION_VECTORS = np.array(
        [
            [0, 0],
            [0, -1],
            [1, -1],
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
        ],
        dtype=np.float16,
    )

    DIRECTION_VECTORS[1:] /= np.linalg.norm(
        DIRECTION_VECTORS[1:], ord=2, axis=1, keepdims=True
    )

    def __init__(self, frame_per_action, number_of_savestates, actions_per_savestate):
        self.frame_per_action = frame_per_action
        self.number_of_savestates = number_of_savestates
        self.actions_per_savestates = actions_per_savestate

    def __call__(self, handler, run_metadata):
        self._register_handler(handler)

        # temporally ordered savestates
        # when the last savestate of the deque is used the first is discarded
        # and put back in the queue
        savestates = deque(
            self.handler.savestate_object() for i in range(self.number_of_savestates)
        )
        savestates_state_id = WeakKeyDictionary()
        dead_ends = set()

        state_id = None
        savestate_idx = 0
        savestate = savestates[savestate_idx]
        savestates_state_id[savestate] = savestate_id = state_id
        actions_since_savestate = 0

        handler.emu_poweron()
        self._play_starting_input_sequence()
        self.handler.savestate_save(savestate)
        self.handler.savestate_persist(savestate)

        fire_frame = True

        while True:
            try:
                direction, state_id = self._get_next_direction(state_id, dead_ends)
            except DeadEnd as dead_end:
                dead_ends -= set(dead_end.dead_directions)
                dead_ends.add(state_id)

                if savestate_id == state_id:
                    if savestate_idx == 0:
                        raise
                    savestate_idx -= 1
                    savestate = savestates[savestate_idx]
                    savestate_id = savestates_state_id[savestate]

                self.handler.savestate_load(savestate)
                state_id = savestate_id
                actions_since_savestate = 0
                continue

            inputs = self._get_direction_as_joypad_inputs(direction)

            for i in range(self.frame_per_action):
                fire_frame = self._frameadvance_with_inputs_and_autofire(
                    fire_frame, inputs
                )

                if reset := self._reset_condition():
                    break

            actions_since_savestate += 1

            if reset:
                dead_ends.add(state_id)
                self.handler.savestate_load(savestate)
                state_id = savestate_id
                reset = False
                actions_since_savestate = 0
                continue

            if actions_since_savestate < self.actions_per_savestates:
                continue

            if savestate_idx + 1 == len(savestates):
                savestate = savestates.popleft()
                savestates.append(savestate)
            else:
                savestate_idx += 1
                savestate = savestates[savestate_idx]

            savestates_state_id[savestate] = savestate_id = state_id
            self.handler.savestate_save(savestate)
            self.handler.savestate_persist(savestate)

            actions_since_savestate = 0

    def _register_handler(self, handler):
        self.handler = handler

    def _play_starting_input_sequence(self):
        for i in range(2):
            self.handler.emu_nframeadvance(np.uint8(60).tobytes())
            self.handler.joypad_set(P1, "start")

        while True:
            if self._is_alive():
                break
            self.handler.emu_frameadvance()

    def _reset_condition(self):
        return not self._is_alive()

    def _is_alive(self):
        return (
            self.handler.memory_readbyterange(
                self.IS_ALIVE_ADDRESS,
                self.ONE,
                self.bONE,
            )[0]
            == 1
        )

    def _xy_coordinates(self):
        return (
            self.handler.memory_readbyterange(
                self.X_AXIS_ADDRESS,
                self.ONE,
                self.bONE,
            )[0],
            self.handler.memory_readbyterange(
                self.Y_AXIS_ADDRESS,
                self.ONE,
                self.bONE,
            )[0],
        )

    def _get_next_direction(self, state_id, dead_ends):
        x, y = self._xy_coordinates()

        direction_toward_balance = np.array(
            (self.X_AXIS_BALANCE - x, self.Y_AXIS_BALANCE - y), dtype=np.int16
        )

        preferred_directions = (
            -np.matmul(self.DIRECTION_VECTORS, direction_toward_balance)
        ).argsort(kind="stable")

        direction_ids = []
        for direction in preferred_directions:
            direction_id = hash((state_id, int(direction)))
            direction_ids.append(direction_id)
            if direction_id not in dead_ends:
                return int(direction), direction_id

        raise DeadEnd(direction_ids)

    def _get_direction_as_joypad_inputs(self, direction):
        if direction == 0:
            return []

        inputs = []
        if (direction < 3) or direction == 8:
            inputs.append("up")
        if (direction > 1) and (direction < 5):
            inputs.append("right")
        if (direction > 3) and (direction < 7):
            inputs.append("down")
        if (direction > 5) and (direction < 9):
            inputs.append("left")

        return inputs

    def _frameadvance_with_inputs_and_autofire(self, fire, inputs):
        if fire:
            self.handler.joypad_set(P1, "A", *inputs)
        else:
            self.handler.joypad_set(P1, *inputs)

        self.handler.emu_frameadvance()

        return not fire


def gradius_loop_fn_init():
    return GradiusLoopFn(FRAME_PER_ACTION, NUMBER_OF_SAVESTATES, ACTIONS_PER_SAVESTATES)


class DeadEnd(Exception):
    def __init__(self, dead_directions):
        self.dead_directions = dead_directions


if __name__ == "__main__":

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
    ).run()
