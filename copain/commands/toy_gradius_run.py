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
from copain.copain_driver import P1


class GradiusLoopFn:
    MEMORY_BYTERANGE_START = np.uint16(0).tobytes()
    MEMORY_BYTERANGE_LENGTH = np.uint16(0)
    MEMORY_BYTERANGE_BLENGTH = MEMORY_BYTERANGE_LENGTH.tobytes()
    ACTION_PER_SECOND = 5

    IS_ALIVE_ADDRESS = 0x0100

    def __call__(self, handler, run_metadata):
        self._register_handler(handler)

        while True:
            handler.emu_poweron()
            self._starting_sequence(handler)
            gamestate = self._get_gamestate(handler)
            inputs = self._get_inputs(gamestate)

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
            self.MEMORY_BYTERANGE_BLENGTH,
            self.MEMORY_BYTERANGE_LENGTH,
        )

    def _get_inputs(self, gamestate):
        raise NotImplementedError


if __name__ == "__main__":

    copain = CopainRun(
        rom_path=ROM_PATH,
        rom_hash=ROM_HASH,
        loop_fn=GradiusLoopFn(),
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
