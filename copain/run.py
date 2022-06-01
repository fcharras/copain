import os
import subprocess
from pathlib import Path
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
from socketserver import UnixStreamServer, ThreadingMixIn

from pyvirtualdisplay import Display

from copain.copain_driver import _CopainSocketHandler, _CopainLoopFn


class _SequentialSocketServer(ThreadingMixIn, UnixStreamServer):
    pass


class CopainRun:
    def __init__(
        self,
        rom_path,
        rom_hash,
        loop_fn_init,
        enable_game_genie=False,
        display_fceux_gui=True,
        enable_sound=True,
        speedmode=None,
        render_sprites=None,
        render_background=None,
        tmp_folder="/tmp",
        fceux_executable="/usr/bin/fceux",
    ):
        self.rom_path = rom_path
        self.rom_hash = rom_hash
        self.loop_fn_init = loop_fn_init

        self.enable_game_genie = enable_game_genie
        self.display_fceux_gui = display_fceux_gui
        self.enable_sound = enable_sound
        self.speedmode = speedmode
        self.render_sprites = render_sprites
        self.render_background = render_background
        self.tmp_folder = tmp_folder
        self.fceux_executable = fceux_executable

    def run(self):

        with ExitStack() as stack:
            run_code_folder = Path(__file__).parent.resolve()
            lua_script = os.path.join(run_code_folder, "run.lua")

            tmp_folder = stack.enter_context(
                TemporaryDirectory(dir=self.tmp_folder, prefix="copain_")
            )
            print(f"Created runtime directory {tmp_folder}...")

            socket_file = os.path.join(tmp_folder, "copain_socket")

            copain_loop_fn = _CopainLoopFn(
                self.speedmode,
                self.render_sprites,
                self.render_background,
                self.rom_path,
                self.rom_hash,
                self.loop_fn_init,
            )

            class _SocketHandler(_CopainSocketHandler):
                def handle(self):
                    return copain_loop_fn(self)

            server = stack.enter_context(_SequentialSocketServer(socket_file, _SocketHandler))
            print(f"Started the socket server at {socket_file}...")
            print("Starting the socket server thread...")

            server_executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
            server_thread = server_executor.submit(server.serve_forever)

            env = dict(os.environb)
            env[b"COPAIN_SOCKET_SERVER_FILE_PATH"] = socket_file.encode()

            start_command = [
                self.fceux_executable,
                "--loadlua",
                lua_script,
                self.rom_path,
            ]

            if self.display_fceux_gui:
                start_command.extend(
                    ["-s", "1" if self.enable_sound else "0"]
                )
                start_command.extend(
                    ["-g", "1" if self.enable_game_genie else "0"]
                )
                print(
                    "Starting visible fceux instance with command \n %s..."
                    % " ".join(start_command)
                )

            else:
                start_command.extend(
                    ["-g", "1" if self.enable_game_genie else "0"]
                )
                start_command.extend(["-s", "0"])
                print(
                    "Starting a background fceux instance with command \n %s..."
                    % (" ".join(start_command))
                )
                stack.enter_context(
                    Display(use_xauth=True, visible=False, size=(1, 1))
                )

            env[b"DISPLAY"] = os.environb[b"DISPLAY"]
            subprocess.Popen(
                start_command,
                env=env
                ).wait()

            server.shutdown()
            server.server_close()
            server_thread.result()
