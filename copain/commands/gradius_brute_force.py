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

NB_FAILS_BEFORE_POSITION_CHANGE = 512

PREFERRED_POSITIONS = (
    (48, 96),  # default position when game starts
    (48, 60),  # top half of screen
    (48, 148),  # bottom half of screen
    (240, 96),  # right end of screen, necessary for end boss
)


from collections import deque
from weakref import ref, WeakKeyDictionary

import numpy as np

from copain import CopainRun
from copain.copain_driver import P1


PREFERRED_POSITIONS = tuple((np.int16(x), np.int16(y)) for x, y in PREFERRED_POSITIONS)


class GameState:
    def __init__(self, action_space_size, parent=None):
        self.transitions = dict()
        self.nb_dead_actions = 0
        self.is_dead_end = False
        self.action_space_size = action_space_size
        self.parent = ref(parent) if parent is not None else None
        self.depth = 0 if parent is None else (parent.depth + 1)

    def mark_dead_end(self):
        self.is_dead_end = True
        parent = self.parent
        del self.transitions, self.parent
        if parent is None:
            raise RuntimeError
        parent = parent()
        parent.nb_dead_actions += 1
        if parent.nb_dead_actions == parent.action_space_size:
            parent.mark_dead_end()


class GradiusLoopFn:
    ACTION_SPACE_SIZE = 9

    ONE = np.uint16(1)
    bONE = ONE.tobytes()
    IS_ALIVE_ADDRESS = np.uint16(0x100).tobytes()

    X_AXIS_ADDRESS = np.uint16(0x360).tobytes()  # range 16 (left) - 240 (right)
    Y_AXIS_ADDRESS = np.uint16(0x320).tobytes()  # range 16 (top) - 192 (right)

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

    def __init__(
        self,
        frame_per_action,
        number_of_savestates,
        actions_per_savestate,
        preferred_positions,
        nb_fails_before_position_change,
    ):
        self.frame_per_action = frame_per_action
        self.number_of_savestates = number_of_savestates
        self.actions_per_savestates = actions_per_savestate
        self.preferred_positions = preferred_positions
        self.nb_preferred_positions = len(preferred_positions)
        self.nb_fails_before_position_change = nb_fails_before_position_change

    def __call__(self, handler, run_metadata):
        self._register_handler(handler)

        # temporally ordered savestates
        # when the last savestate of the deque is used the first is discarded
        # and put back in the queue
        savestates = deque(
            self.handler.savestate_object() for i in range(self.number_of_savestates)
        )
        # weakref dictionary to store the states linked to those savestates without
        # worrying about cleaning if said savestates are deleted
        savestates_states = WeakKeyDictionary()

        # Notes about reference managements: we maintain a tree of all visited
        # states with a tight reference management to clean the nodes as soon
        # as the states are not reachable anymore from the loadable savestates.

        # To this purpose we explicitly use only three strong references:
        # - the reference to the root state which is the state of the oldest
        # savestate (root_state)
        # - the reference to the savestate currently used in case of failure
        # (gamestate_state)
        # - the reference to the current state (state)
        # The states hold themselves strong references to subsequent states.
        # All other references are weak references.
        savestate_idx = 0
        savestate = savestates[savestate_idx]
        savestate_state = root_state = state = GameState(self.ACTION_SPACE_SIZE)
        savestates_states[savestate] = ref(state)
        actions_since_savestate = 0

        max_depth = 0
        nb_fails_without_improvements = 0
        preferred_position = 0
        preference_reset_depth = 0

        handler.emu_poweron()
        self._play_starting_input_sequence()
        self.handler.savestate_save(savestate)
        self.handler.savestate_persist(savestate)

        fire_frame = True

        while True:
            direction, state = self._get_next_direction(state, preferred_position)
            inputs = self._get_direction_as_joypad_inputs(direction)

            for i in range(self.frame_per_action):
                fire_frame = self._frameadvance_with_inputs_and_autofire(
                    fire_frame, inputs
                )

                if reset := self._reset_condition():
                    break

            if reset:
                state.mark_dead_end()

                current_depth = state.depth - 1
                if current_depth > max_depth:
                    max_depth = current_depth
                    nb_fails_without_improvements = 0
                else:
                    nb_fails_without_improvements += 1

                if nb_fails_without_improvements > self.nb_fails_before_position_change:
                    savestate_idx = 0
                    savestate = savestates[savestate_idx]
                    savestate_state = savestates_states[savestate]()
                    preference_reset_depth = max_depth
                    max_depth = savestate_state.depth
                    nb_fails_without_improvements = 0
                    preferred_position = (
                        preferred_position + 1
                    ) % self.nb_preferred_positions

                while savestate_state.is_dead_end:
                    if savestate_idx == 0:
                        raise RuntimeError
                    savestate_idx -= 1
                    savestate = savestates[savestate_idx]
                    savestate_state = savestates_states[savestate]()

                self.handler.savestate_load(savestate)
                state = savestate_state
                actions_since_savestate = 0
                continue

            actions_since_savestate += 1
            if state.depth > preference_reset_depth:
                preferred_position = 0
                preference_reset_depth = 0

            if actions_since_savestate < self.actions_per_savestates:
                continue

            if savestate_idx + 1 == len(savestates):
                savestate = savestates.popleft()
                savestates.append(savestate)
                root_state = savestates_states[savestates[0]]()  # noqa
            else:
                savestate_idx += 1
                savestate = savestates[savestate_idx]

            savestate_state = state
            savestates_states[savestate] = ref(state)
            self.handler.savestate_save(savestate)
            self.handler.savestate_persist(savestate)

            actions_since_savestate = 0

    def _play_starting_input_sequence(self):
        for i in range(2):
            self.handler.emu_nframeadvance(np.uint8(60).tobytes())
            self.handler.joypad_set(P1, "start")

        while True:
            if self._is_alive():
                break
            self.handler.emu_frameadvance()

    def _get_next_direction(self, state, preferred_position):
        x, y = self._xy_coordinates()

        preferred_x, preferred_y = self.preferred_positions[preferred_position]

        direction_toward_preferred_position = np.array(
            (preferred_x - np.int16(x), preferred_y - np.int16(y)), dtype=np.int16
        )

        preferred_directions = (
            -np.matmul(self.DIRECTION_VECTORS, direction_toward_preferred_position)
        ).argsort(kind="stable")

        for direction in preferred_directions:
            if not (
                direction_state := state.transitions.setdefault(
                    direction, GameState(self.ACTION_SPACE_SIZE, parent=state)
                )
            ).is_dead_end:
                return direction, direction_state

        raise RuntimeError

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

    def _reset_condition(self):
        # TODO: also detect end of game ?
        return not self._is_alive()

    def _register_handler(self, handler):
        self.handler = handler


def gradius_loop_fn_init():
    return GradiusLoopFn(
        FRAME_PER_ACTION,
        NUMBER_OF_SAVESTATES,
        ACTIONS_PER_SAVESTATES,
        PREFERRED_POSITIONS,
        NB_FAILS_BEFORE_POSITION_CHANGE,
    )


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
