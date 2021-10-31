import os
import subprocess
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from socketserver import UnixStreamServer, BaseRequestHandler, ThreadingMixIn


import numpy as np


class _ActionRegistry:
    def __init__(self):
        self._one = np.uint16(1)
        self._next_registry_code = self._one

    def register_action(self, action):
        code = self._next_registry_code
        self._next_registry_code += self._one

        def initiatored_action(driver, *args, **kwargs):
            driver.request.sendall(code)
            return action(driver, *args, **kwargs)

        return initiatored_action


class _FceuxDriver:
    """Mirror the lua functions in fceux. To enable initiating the actions
    through the socket the actions must be numbered. The number we give to the
    functions is infered from the order in which they are defined. Inserting
    a function in position e.g 3 will change the number of all the subsequent
    functions. The lua side of this protocol must then be adapted accordingly.

    Not all available function in lua have been implemented yet. It's easy to
    implement other functions if necessary.
    """

    register_action = _ActionRegistry().register_action

    """emu.* namespace"""

    @register_action  # 1
    def emu_poweron(self):
        return

    _SPEEDMODE_CODES = dict(
        normal=np.uint8(1).tobytes(),
        nothrottle=np.uint8(2).tobytes(),
        turbo=np.uint8(3).tobytes(),
        maximum=np.uint8(4).tobytes(),
    )

    @register_action  # 2
    def emu_speedmode(self, speedmode):
        self.request.sendall(self._SPEEDMODE_CODES[speedmode])

    @register_action  # 3
    def emu_frameadvance(self):
        return

    @register_action  # 4
    def emu_nframeadvance(self, n):
        """n expected to be of type bytes, of length 1, encoding a uint8"""
        self.request.sendall(n)

    @register_action  # 5
    def emu_setrenderplanes(self, sprites, background):
        """sprites and background expected to be of type bytes, of length 1,
        encoding a uint8 either equal to 0 or 1"""
        self.request.sendall(sprites + background)

    @register_action  # 6
    def emu_loadrom(self, path):
        """path is expected to be of type bytes of length at most 2**16"""
        self.request.sendall(np.uint16(len(path)).tobytes())
        self.request.sendall(path)

    @register_action  # 7
    def emu_exit(self):
        return

    """rom.* namespace"""

    _HASH_TYPE_CODES = dict(md5=np.uint8(1).tobytes(), base64=np.uint8(2).tobytes())
    _EXPECTED_HASH_LENGTH = dict(md5=32, base64=31)  # TODO: find good values

    @register_action  # 8
    def rom_gethash(self, hash_type):
        self.request.sendall(self._HASH_TYPE_CODES[hash_type])
        return self.request.recv(self._EXPECTED_HASH_LENGTH[hash_type]).decode()

    """memory.* namespace"""

    MEMORY_ADDRESS_TYPE = np.uint8

    @register_action  # 9
    def memory_readbyterange(self, address, length):
        """address is expected to be of type bytes, length a numpy unsigned int of the same size"""
        self.request.sendall(address + length.tobytes())
        return np.frombuffer(self.request.recv(length), dtype=self.MEMORY_ADDRESS_TYPE)

    """ppu.* namespace"""

    PPU_ADDRESS_TYPE = np.uint8

    @register_action  # 10
    def ppu_readbyterange(self, address, length):
        """address is expected to be of type bytes, length a numpy unsigned int of the same size"""
        self.request.sendall(address + length.tobytes())
        return np.frombuffer(self.request.recv(length), dtype=self.PPU_ADDRESS_TYPE)

    """joypad.* namespace"""

    @register_action  # 11
    def joypad_set(self, inputs):
        """inputs have type dict with keys up, down, left, right, A, B, start, select, and values 0 (button held) or 1 (button released) of type np.uint8
        TODO: would it be better to choose a more compact encoding for socket transmission ? e.g only one entry to encode all possible 9 directions ?"""
        self.request.sendall(
            inputs["player"]
            + inputs["up"]
            + inputs["down"]
            + inputs["left"]
            + inputs["right"]
            + inputs["A"]
            + inputs["B"]
            + inputs["start"]
            + inputs["select"]
        )

    """misc"""

    @register_action  # 12
    def get_runner_id(self):
        return np.frombuffer(self.request.recv(2), dtype=np.uint16)[0]


@dataclass(frozen=True)
class CopainRunMetadata:
    _VISIBLE_RUNNER_ID = 1

    rom_path: str
    rom_hash: str
    runner_id: int
    total_nb_runners: int
    is_visible_runner: bool = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self, "is_visible_runner", self.runner_id == self._VISIBLE_RUNNER_ID
        )


class _CopainLoopFn:
    def __init__(
        self,
        display_visible_runner,
        total_nb_runners,
        visible_speedmode,
        visible_render_sprites,
        visible_render_background,
        rom_path,
        rom_hash,
        loop_fn,
    ):
        self.display_visible_runner = display_visible_runner
        self.total_nb_runners = total_nb_runners
        self.visible_speedmode = visible_speedmode
        self.visible_render_sprites = visible_render_sprites
        self.visible_render_background = visible_render_background
        self.rom_path = rom_path
        self.rom_hash = rom_hash
        self.loop_fn = loop_fn

    def __call__(self, handler):
        if not handler.request.getblocking():
            raise RuntimeError("Expected a socket in blocking mode.")

        runner_id = handler.get_runner_id()

        run_metadata = CopainRunMetadata(
            rom_path=self.rom_path,
            rom_hash=self.rom_hash,
            runner_id=runner_id,
            total_nb_runners=self.total_nb_runners,
        )
        is_visible_runner = run_metadata.is_visible_runner

        if is_visible_runner and self.display_visible_runner:
            if self.visible_speedmode is not None:
                speedmode = self.visible_speedmode
            else:
                speedmode = "maximum" if self.total_nb_runners == 1 else "normal"

            visible_render_sprites = (
                self.visible_render_sprites is None
            ) or self.visible_render_sprites

            visible_render_background = (
                self.visible_render_background is None
            ) or self.visible_render_background

        else:
            speedmode = "maximum"
            visible_render_sprites = False
            visible_render_background = False

        handler.emu_speedmode(speedmode)
        handler.emu_setrenderplanes(
            sprites=b"\x01" if visible_render_sprites else b"\x00",
            background=b"\x01" if visible_render_background else b"\x00",
        )

        actual_hash = handler.rom_gethash("md5")
        if actual_hash != self.rom_hash:
            raise RuntimeError(
                f"The ROM that has been loaded at {self.rom_path} has md5 hash "
                f"{actual_hash} but expected hash {self.rom_hash}"
            )

        loop_fn_signature = inspect.signature(self.loop_fn).parameters
        pass_run_metadata = ("run_metadata" in loop_fn_signature) and not any(
            kind is loop_fn_signature["run_metadata"].kind
            for kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        )

        return self.loop_fn(
            handler,
            **(dict(run_metadata=run_metadata) if pass_run_metadata else dict()),
        )


class _CopainSequentialSocketServer(ThreadingMixIn, UnixStreamServer):
    pass


class _CopainThreadedSocketServer(ThreadingMixIn, UnixStreamServer):
    pass


class CopainRun:
    def __init__(
        self,
        rom_path,
        rom_hash,
        loop_fn,
        threaded_socket=False,
        num_runners=1,
        enable_game_genie=False,
        display_fceux_gui=True,
        visible_enable_sound=True,
        visible_speedmode=None,
        visible_render_sprites=None,
        visible_render_background=None,
        tmp_folder="/tmp",
        fceux_executable="/usr/bin/fceux",
    ):
        self.rom_path = rom_path
        self.rom_hash = rom_hash
        self.loop_fn = loop_fn
        self.threaded_socket = threaded_socket
        self.num_runners = num_runners
        self.enable_game_genie = enable_game_genie
        self.display_fceux_gui = display_fceux_gui
        self.visible_enable_sound = visible_enable_sound
        self.visible_speedmode = visible_speedmode
        self.visible_render_sprites = visible_render_sprites
        self.visible_render_background = visible_render_background
        self.tmp_folder = tmp_folder
        self.fceux_executable = fceux_executable

    def run(self):
        max_nb_runners = np.iinfo(np.uint16).max
        if self.num_runners > max_nb_runners:
            raise ValueError(
                f"num_runners is expected to be below "
                f"{max_nb_runners}, got {self.num_runners}"
            )

        with ExitStack() as stack:
            run_code_folder = Path(__file__).parent.resolve()
            lua_script = os.path.join(run_code_folder, "copain_main.lua")

            tmp_folder = stack.enter_context(
                TemporaryDirectory(dir=self.tmp_folder, prefix="copain_")
            )
            print(f"Created runtime directory {tmp_folder}...")

            socket_file = os.path.join(tmp_folder, "copain_socket")

            env = os.environb.copy()
            env["COPAIN_SOCKET_SERVER_FILE_PATH"] = socket_file.encode()

            base_start_command = [
                self.fceux_executable,
                "--loadlua",
                lua_script,
                self.rom_path,
            ]

            normal_start_command = base_start_command.copy()
            if not self.display_fceux_gui:
                normal_start_command.append("--no-gui")
            normal_start_command.extend(
                ["-s", "1" if self.visible_enable_sound else "0"]
            )
            normal_start_command.extend(["-g", "1" if self.enable_game_genie else "0"])

            background_start_command = base_start_command.copy()
            # for now it's bugged, it triggers a segfault
            # background_start_command.append("--no-gui")
            background_start_command.extend(
                ["-g", "1" if self.enable_game_genie else "0"]
            )
            background_start_command.extend(["-s", "0"])

            print(
                "Starting visible fceux instance with command \n %s..."
                % " ".join(normal_start_command)
            )
            print(
                "Starting %s background fceux instances with command \n %s..."
                % (self.num_runners - 1, " ".join(background_start_command))
            )

            runs = []
            for i in range(1, self.num_runners + 1):
                i8 = np.uint8(i)
                i16 = np.uint16(i)
                i = i8 if (i8 == i16) else i16
                env["COPAIN_RUN_ID"] = i.tobytes()
                runs.append(
                    subprocess.Popen(
                        normal_start_command if (i == 1) else background_start_command,
                        env=env,
                    )
                )

            del runs

            server_type = (
                _CopainThreadedSocketServer
                if self.threaded_socket
                else _CopainSequentialSocketServer
            )

            copain_loop_fn = _CopainLoopFn(
                self.display_fceux_gui,
                self.num_runners,
                self.visible_speedmode,
                self.visible_render_sprites,
                self.visible_render_background,
                self.rom_path,
                self.rom_hash,
                self.loop_fn,
            )

            class _CopainSocketHandler(BaseRequestHandler, _FceuxDriver):
                def handle(self):
                    return copain_loop_fn(self)

            server = stack.enter_context(server_type(socket_file, _CopainSocketHandler))

            print(f"Started the socket server at {socket_file}...")

            print("Starting the socket server thread...")
            server.serve_forever()


if __name__ == "__main__":

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
    THREADED_SOCKET = False

    def gradius_loop_fn(handler, run_metadata):
        false = np.uint8(0).tobytes()
        true = np.uint8(1).tobytes()
        press_start = dict(
            player=true,
            up=false,
            down=false,
            left=false,
            right=false,
            A=false,
            B=false,
            start=true,
            select=false,
        )

        for i in range(2):
            handler.emu_nframeadvance(np.uint8(60).tobytes())
            handler.joypad_set(press_start)

        while True:
            vic_is_alive = (
                handler.memory_readbyterange(np.uint16(0).tobytes(), np.uint16(0x8000))[
                    0x0100
                ]
                == 1
            )
            if vic_is_alive:
                print("Vic has materialized !")
                break
            handler.emu_frameadvance()

        handler.emu_nframeadvance(np.uint8(600).tobytes())

    copain = CopainRun(
        rom_path=ROM_PATH,
        rom_hash=ROM_HASH,
        loop_fn=gradius_loop_fn,
        threaded_socket=THREADED_SOCKET,
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
