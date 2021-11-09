import os
import subprocess
from pathlib import Path
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from socketserver import UnixStreamServer, ThreadingMixIn
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyvirtualdisplay import Display

from copain.copain_driver import _CopainSocketHandler, _CopainLoopFn


class _SequentialSocketServer(ThreadingMixIn, UnixStreamServer):
    pass


class _ThreadedSocketServer(ThreadingMixIn, UnixStreamServer):
    pass


class CopainRun:
    def __init__(
        self,
        rom_path,
        rom_hash,
        loop_fn_init,
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
        copain_ai=None,
        threaded_ai=True,
    ):
        self.rom_path = rom_path
        self.rom_hash = rom_hash
        self.loop_fn_init = loop_fn_init
        self.threaded_socket = threaded_socket

        if num_runners > 1 and not threaded_requests:
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
        self.copain_ai = copain_ai
        self.threaded_ai = threaded_ai

    def run(self):

        with ExitStack() as stack:
            run_code_folder = Path(__file__).parent.resolve()
            lua_script = os.path.join(run_code_folder, "run.lua")

            tmp_folder = stack.enter_context(
                TemporaryDirectory(dir=self.tmp_folder, prefix="copain_")
            )
            print(f"Created runtime directory {tmp_folder}...")

            socket_file = os.path.join(tmp_folder, "copain_socket")

            env = dict(os.environb)
            env[b"COPAIN_SOCKET_SERVER_FILE_PATH"] = socket_file.encode()

            base_start_command = [
                self.fceux_executable,
                "--loadlua",
                lua_script,
                self.rom_path,
            ]

            visible_start_command = None
            background_start_command = None

            if self.display_fceux_gui:
                visible_start_command = base_start_command.copy()

                visible_start_command.extend(
                    ["-s", "1" if self.visible_enable_sound else "0"]
                )
                visible_start_command.extend(
                    ["-g", "1" if self.enable_game_genie else "0"]
                )
                print(
                    "Starting visible fceux instance with command \n %s..."
                    % " ".join(visible_start_command)
                )

            if not self.display_fceux_gui or self.num_runners > 1:
                background_start_command = base_start_command.copy()
                background_start_command.extend(
                    ["-g", "1" if self.enable_game_genie else "0"]
                )
                background_start_command.extend(["-s", "0"])

                nb_background_instances = (
                    (self.num_runners - 1)
                    if self.display_fceux_gui
                    else self.num_runners
                )
                print(
                    "Starting %s background fceux instances with command \n %s..."
                    % (nb_background_instances, " ".join(background_start_command))
                )

            runs = []
            for i in range(1, self.num_runners + 1):
                command = background_start_command

                if i == 1 and self.display_fceux_gui:
                    command = visible_start_command

                elif i == 1 or (self.display_fceux_gui and i == 2):
                    stack.enter_context(
                        Display(use_xauth=True, visible=False, size=(1, 1))
                    )

                i8 = np.uint8(i)
                i16 = np.uint16(i)
                i = i8 if (i8 == i16) else i16
                env[b"COPAIN_RUN_ID"] = i.tobytes()
                env[b"DISPLAY"] = os.environb[b"DISPLAY"]
                runs.append(
                    subprocess.Popen(
                        command,
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
                self.loop_fn_init,
                self.copain_ai,
            )

            class _SocketHandler(_CopainSocketHandler):
                def handle(self):
                    return copain_loop_fn(self)

            server = stack.enter_context(server_type(socket_file, _SocketHandler))

            if self.copain_ai is not None and self.threaded_ai:
                ai_executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
                copain_ai_thread = ai_executor.submit(self.copain_ai.serve_forever)

            print(f"Started the socket server at {socket_file}...")

            print("Starting the socket server thread...")
            if self.threaded_socket:
                server_executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
                server_thread = server_executor.submit(server.serve_forever)
            else:
                server.serve_forever()

            try:
                for run in runs:
                    try:
                        run.wait()
                    except BaseException:
                        break
            finally:
                for run in runs:
                    if run.poll() is None:
                        run.kill()

                server.shutdown()
                server.server_close()

                if self.threaded_socket:
                    server_thread.result()
                    del server_thread

                if self.copain_ai is not None:
                    self.copain_ai.shutdown()
                    if self.threaded_ai:
                        copain_ai_thread.result()
