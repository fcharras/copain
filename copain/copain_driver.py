import inspect
from collections import deque
from dataclasses import dataclass, field
from socketserver import BaseRequestHandler


import numpy as np


BFALSE = np.uint8(0).tobytes()
BTRUE = np.uint8(1).tobytes()

P1 = np.uint8(1).tobytes()
P2 = np.uint8(2).tobytes()

IDLE_JOYPAD = dict(
    up=BFALSE,
    down=BFALSE,
    left=BFALSE,
    right=BFALSE,
    A=BFALSE,
    B=BFALSE,
    start=BFALSE,
    select=BFALSE,
)

DIRECTIONS = ["up", "down", "left", "right"]


class _ActionRegistry:
    def __init__(self):
        self._one = np.uint16(1)
        self._next_registry_code = self._one

    def register_action(self, action):
        code = self._next_registry_code
        self._next_registry_code += self._one

        def initiatored_action(driver, *args, **kwargs):
            driver.request.sendall(code)
            try:
                return action(driver, *args, **kwargs)
            finally:
                driver._gc()

        return initiatored_action


class _Savestate:
    GC_QUEUE = deque()
    ID_LENGTH = 16

    def __init__(self):
        self._id = id(self).to_bytes(self.ID_LENGTH, "big")

    def get_savestate_id(self):
        return self._id

    def __del__(self):
        self.GC_QUEUE.append(self._id)


class _Driver:
    """Mirror the lua functions in fceux. To enable initiating the actions
    through the socket the actions must be numbered. The number we give to the
    functions is infered from the order in which they are defined. Inserting
    a function in position e.g 3 will change the number of all the subsequent
    functions. The lua side of this protocol must then be adapted accordingly.

    Not all available function in lua have been implemented yet. It's easy to
    implement other functions if necessary.

    TODO: on some platforms there might be issue arising from endianness differences in python
    and lua ?
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
        self._send_load(path)

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
    def memory_readbyterange(self, address, length, blength):
        """address is expected to be of type bytes, length a numpy unsigned int of the same size, blength the same array but bytes encoded"""
        self.request.sendall(address + blength)
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
    def joypad_set(self, player, *inputs):
        """inputs have type dict with keys up, down, left, right, A, B, start, select, and values 0 (button held) or 1 (button released) of type np.uint8
        TODO: would it be better to choose a more compact encoding for socket transmission ? e.g only one entry to encode all possible 9 directions ?"""
        res = IDLE_JOYPAD.copy()
        for press in inputs:
            res[press] = BTRUE

        res["player"] = player

        self.request.sendall(b"".join((
            res["player"],
            res["up"],
            res["down"],
            res["left"],
            res["right"],
            res["A"],
            res["B"],
            res["start"],
            res["select"]
        )))

    """savestate.* namespace"""

    _EXPECTED_SAVESTATE_ID_LENGTH = 16
    if _EXPECTED_SAVESTATE_ID_LENGTH != _Savestate.ID_LENGTH:
        raise RuntimeError(
            "The length of a savestate id is inconsistent: expected %s, got %s ."
            % (_EXPECTED_SAVESTATE_ID_LENGTH, _Savestate.ID_LENGTH)
        )

    def savestate_object(self, slot: bytes = None):
        """if not None, slot is expected to be of type bytes, of length 1, encoding a uint8"""

        if slot is not None:
            self._savestate_slotted_object(slot)
        else:
            self._savestate_anonymous_object()

        savestate_object = _Savestate()
        self._send_savestate_id(savestate_object)
        return savestate_object

    @register_action  # 12
    def _savestate_slotted_object(self, slot: bytes):
        self.request.sendall(slot)

    @register_action  # 13
    def _savestate_anonymous_object(self):
        return

    @register_action  # 14
    def savestate_save(self, savestate_object: _Savestate):
        self._send_savestate_id(savestate_object)

    @register_action  # 15
    def savestate_load(self, savestate_object: _Savestate):
        self._send_savestate_id(savestate_object)

    @register_action  # 16
    def savestate_persist(self, savestate_object: _Savestate):
        self._send_savestate_id(savestate_object)

    @register_action  # 17  (for internal use only)
    def _savestate_gc(self, savestate_id):
        self.request.sendall(savestate_id)

    def _send_savestate_id(self, savestate_object):
        self.request.sendall(savestate_object.get_savestate_id())

    """movie"""

    @register_action  # 18
    def movie_play(self, path):
        """path is expected to be of type bytes of length at most 2**16"""
        self._send_load(path)

    @register_action  # 19
    def movie_record(self, path, save_type, author):
        """path and authors are expected to be of type bytes of length at most 2**16
        save_type expected to be of type bytes, of length 1, encoding a uint8 0, 1 or 2"""
        self._send_load(path)
        self.request.sendall(save_type)
        self._send_load(author)

    @register_action  # 20
    def movie_stop(self):
        return

    @register_action  # 21
    def movie_rerecordcounting(self, counting):
        self.request.sendall(BTRUE if counting else BFALSE)

    @register_action  # 22
    def movie_rerecordcount(self):
        return np.frombuffer(self.request.recv(8), dtype=np.uint64)[0]


    """misc"""

    @register_action  # 23
    def get_runner_id(self):
        return np.frombuffer(self.request.recv(2), dtype=np.uint16)[0]

    """internal only"""

    def _gc(self):
        self._savestates_gc()

    def _savestates_gc(self):
        gc_queue = _Savestate.GC_QUEUE
        while gc_queue:
            savestate_id = gc_queue.popleft()
            self._savestate_gc(savestate_id)

    def _send_load(self, load):
        self.request.sendall(np.uint16(len(load)).tobytes())
        self.request.sendall(load)

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
        loop_fn_init,
        copain_ai,
    ):
        self.display_visible_runner = display_visible_runner
        self.total_nb_runners = total_nb_runners
        self.visible_speedmode = visible_speedmode
        self.visible_render_sprites = visible_render_sprites
        self.visible_render_background = visible_render_background
        self.rom_path = rom_path
        self.rom_hash = rom_hash
        self.loop_fn_init = loop_fn_init
        self.copain_ai = copain_ai

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

        kwargs = dict()

        loop_fn = self.loop_fn_init()

        loop_fn_signature = inspect.signature(loop_fn).parameters
        if ("run_metadata" in loop_fn_signature) and not any(
            kind is loop_fn_signature["run_metadata"].kind
            for kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        ):
            kwargs["run_metadata"] = run_metadata

        pass_copain_ai = ("copain_ai" in loop_fn_signature) and not any(
            kind is loop_fn_signature["copain_ai"].kind
            for kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        )

        if pass_copain_ai is not (self.copain_ai is not None):
            raise ValueError(
                "Expecting an instantiated copain_ai if and only if the loop_fn expects a parameter named copain_ai"
            )

        if pass_copain_ai:
            kwargs["copain_ai"] = self.copain_ai

        return loop_fn(
            handler,
            **kwargs,
        )


class _CopainSocketHandler(BaseRequestHandler, _Driver):
    pass
