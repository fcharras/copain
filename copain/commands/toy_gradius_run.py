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


import numpy as np


from copain import CopainRun
from copain.copain_driver import P1, DIRECTIONS


class GradiusLoopFn:
    MEMORY_BYTERANGE_START = np.uint16(0).tobytes()
    MEMORY_BYTERANGE_LENGTH = np.uint16(0x8000)
    MEMORY_BYTERANGE_BLENGTH = MEMORY_BYTERANGE_LENGTH.tobytes()

    IS_ALIVE_ADDRESS = 0x100

    def __init__(self, action_per_second):
        self.action_per_second = action_per_second
        self._skip_after_input = action_per_second - 1
        self.random = np.random.RandomState()

    def __call__(self, handler, run_metadata):
        self._register_handler(handler)

        while True:
            handler.emu_poweron()
            self._starting_sequence()
            while True:
                gamestate = self._get_gamestate()
                inputs = self._get_inputs(gamestate)
                handler.joypad_set(P1, *inputs)
                handler.emu_frameadvance()
                inputs_directions = [press for press in inputs if press in DIRECTIONS]
                for i in range(self._skip_after_input):
                    gamestate = self._get_gamestate()
                    must_reset = self._reset_condition(gamestate)
                    if must_reset:
                        break
                    handler.joypad_set(P1, *inputs_directions)
                    self.handler.emu_frameadvance()
                if must_reset:
                    break

    def _register_handler(self, handler):
        self.handler = handler

    def _starting_sequence(self):
        for i in range(2):
            self.handler.emu_nframeadvance(np.uint8(60).tobytes())
            self.handler.joypad_set(P1, "start")

        while True:
            if self._is_alive(self._get_gamestate()):
                print("The player has materialized !")
                break
            self.handler.emu_frameadvance()

    def _reset_condition(self, gamestate):
        return not self._is_alive(gamestate)

    def _is_alive(self, gamestate):
        return gamestate[self.IS_ALIVE_ADDRESS] == 1

    def _get_gamestate(self):
        # TODO: include ppu ?
        return self.handler.memory_readbyterange(
            self.MEMORY_BYTERANGE_START,
            self.MEMORY_BYTERANGE_LENGTH,
            self.MEMORY_BYTERANGE_BLENGTH,
        )

    def _get_inputs(self, gamestate):
        # TODO: implement completely random inputs
        direction = self.random.randint(9)
        A = self.random.randint(2)
        B = self.random.randint(2)
        inputs = []
        if A == 1:
            inputs.append("A")
        if B == 1:
            inputs.append("B")
        if (direction < 2) or direction == 7:
            inputs.append("up")
        if (direction > 0) and (direction < 4):
            inputs.append("right")
        if (direction > 2) and (direction < 6):
            inputs.append("down")
        if (direction > 4) and (direction < 8):
            inputs.append("left")
        return inputs


if __name__ == "__main__":

    copain = CopainRun(
        rom_path=ROM_PATH,
        rom_hash=ROM_HASH,
        loop_fn=GradiusLoopFn(5),
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
