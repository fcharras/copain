import os
import subprocess
from pathlib import Path
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from socketserver import UnixStreamServer, ThreadingMixIn
from concurrent.futures import ThreadPoolExecutor


from copain.copain_driver import _CopainSocketHandler, _CopainLoopFn


import numpy as np


class _SequentialSocketServer(ThreadingMixIn, UnixStreamServer):
    pass


class _ThreadedSocketServer(ThreadingMixIn, UnixStreamServer):
    pass


class CopainRun:
    def __init__(
        self,
        rom_path,
        rom_hash,
        loop_fn,
        threaded_socket=True,
        threaded_requests=True,
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

        if num_runners > 1 and not self.threaded_requests:
            raise ValueError(
                "A threaded handling of the requests is mandatory when "
                "num_runners > 1"
            )
        self.threaded_requests = threaded_requests

        max_nb_runners = np.iinfo(np.uint16).max
        if num_runners > max_nb_runners:
            raise ValueError(
                f"num_runners is expected to be below "
                f"{max_nb_runners}, got {self.num_runners}"
            )
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

        with ExitStack() as stack:
            run_code_folder = Path(__file__).parent.resolve()
            lua_script = os.path.join(run_code_folder, "run.lua")

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

            server_type = (
                _ThreadedSocketServer
                if self.threaded_socket
                else _SequentialSocketServer
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

            class _SocketHandler(_CopainSocketHandler):
                def handle(self):
                    return copain_loop_fn(self)

            server = stack.enter_context(server_type(socket_file, _SocketHandler))

            print(f"Started the socket server at {socket_file}...")

            print("Starting the socket server thread...")
            if self.threaded_socket:
                executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
                executor.submit(server.serve_forever)
            else:
                server.serve_forever()

            try:
                for run in runs:
                    try:
                        run.wait()
                    except BaseException:
                        break
            finally:
                server.shutdown()
                server.server_close()
                for run in runs:
                    if run.poll() is None:
                        run.kill()
