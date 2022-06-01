local socket = require "posix.sys.socket"
local unistd = require "posix.unistd"
local struct = require "struct"  -- TODO: replace with string.pack when lua is updated
                                 -- and update the dockerfile accordingly

local driver = { }
local socket_fd = nil


function driver.start ()
    local socket_path = assert(os.getenv("COPAIN_SOCKET_SERVER_FILE_PATH"))
    driver.socket_fd = assert(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0))
    socket_fd = driver.socket_fd
    assert(socket.connect(socket_fd, {family = socket.AF_UNIX, path = socket_path}))
end


function driver.stop ()
    assert(socket.shutdown(socket_fd, socket.SHUT_RDWR))
    assert(unistd.close(socket_fd))
    socket_fd = nil
    driver.socket_fd = nil
end


function driver.recv_action_code ()
    local action_code = assert(socket.recv(socket_fd, 2))
    return action_code
end


driver.actions_registry = {}
local NEXT_REGISTRY_CODE = 1
local function register_action (action)
    driver.actions_registry[NEXT_REGISTRY_CODE] = action
    NEXT_REGISTRY_CODE = NEXT_REGISTRY_CODE + 1
end


-- utils

local function _unpack_load (length, types)
    local sload = assert(socket.recv(socket_fd, length))
    return struct.unpack(types, sload)
end


local function _pack_load (types, ...)
    local sload = struct.pack(types, ...)
    assert(socket.send(socket_fd, sload))
end


local function _recv_bload (length_bytes, length_type)
    local bload = _unpack_load(length_bytes, length_type)
    bload = assert(socket.recv(socket_fd, bload))
    return bload
end


local function _send_bload (bload, length_type)
    length = string.len(bload)
    _pack_load(length_type, length)
    assert(socket.send(socket_fd, bload))
end


-- emu.* namespace

local function emu_poweron ()
    emu.poweron()
end
register_action(emu_poweron)  -- 1


local SPEEDMODE_STR = {"normal", "nothrottle", "turbo", "maximum"}
local function emu_speedmode ()
    local speedmode_code = _unpack_load(1, "B")
    emu.speedmode(SPEEDMODE_STR[speedmode_code])
end
register_action(emu_speedmode)  -- 2


local function emu_frameadvance ()
    emu.frameadvance ()
end
register_action(emu_frameadvance)  -- 3


local function emu_nframeadvance ()
    local n = _unpack_load(1, "B")
    for step=1,n,1 do
        emu.frameadvance ()
    end
end
register_action(emu_nframeadvance)  -- 4


local function emu_setrenderplanes ()
    local sprites, background = _unpack_load(2, "BB")
    sprites = sprites == 1
    background = background == 1
    emu.setrenderplanes(sprites, background)
end
register_action(emu_setrenderplanes)  -- 5


local function emu_loadrom ()
    local path = _recv_bload(2, "H")
    emu.loadrom(path)
end
register_action(emu_loadrom)  -- 6


local function emu_exit ()
    emu.exit()
end
register_action(emu_exit)  -- 7


-- rom.* namespace

local HASHTYPE_STR = {"md5", "base64"}
local function rom_gethash ()
    local hashtype = _unpack_load(1, "B")
    hashtype = HASHTYPE_STR[hashtype]
    assert(socket.send(socket_fd, rom.gethash (hashtype)))
end
register_action(rom_gethash)  -- 8


-- memory.* namespace

local function memory_readbyterange ()
    local address, length = _unpack_load(4, "HH")
    local memory = memory.readbyterange(address, length)
    assert(socket.send(socket_fd, memory))
end
register_action(memory_readbyterange)  -- 9


-- ppu.* namespace

local function ppu_readbyterange ()
    local address, length = _unpack_load(4, "HH")
    local memory = ppu.readbyterange(address, length)
    assert(socket.send(socket_fd, memory))
end
register_action(ppu_readbyterange)  -- 10


-- joypad.* namespace

local function joypad_set()
    local player, up, down, left, right, A, B, start, selec = _unpack_load(9, "BBBBBBBBB")
    joypad.set(player,
        {
            up = (up == 1),
            down = (down == 1),
            left = (left == 1),
            right = (right == 1),
            A = (A == 1),
            B = (B == 1),
            start = (start == 1),
            selec = (selec == 1)
        }
    )
end
register_action(joypad_set)  -- 11


-- savestate.* namespace

driver._savestate_registry = {}

local function savestate_slotted_object ()
    local n = _unpack_load(1, "B")
    local savestate_id = assert(socket.recv(socket_fd, 16))
    driver._savestate_registry[savestate_id] = savestate.object(n)
end
register_action(savestate_slotted_object)  -- 12


local function savestate_anonymous_object ()
    local savestate_id = assert(socket.recv(socket_fd, 16))
    driver._savestate_registry[savestate_id] = savestate.object()
end
register_action(savestate_anonymous_object)  -- 13


local function savestate_save ()
    local savestate_id = assert(socket.recv(socket_fd, 16))
    savestate.save(driver._savestate_registry[savestate_id])
end
register_action(savestate_save)  -- 14


local function savestate_load ()
    local savestate_id = assert(socket.recv(socket_fd, 16))
    savestate.load(driver._savestate_registry[savestate_id])
end
register_action(savestate_load)  -- 15


local function savestate_persist ()
    local savestate_id = assert(socket.recv(socket_fd, 16))
    savestate.persist(driver._savestate_registry[savestate_id])
end
register_action(savestate_persist)  -- 16

local function savestate_gc ()
    local savestate_id = assert(socket.recv(socket_fd, 16))
    driver._savestate_registry[savestate_id] = nil
end
register_action(savestate_gc)  -- 17


-- movie

local function movie_play ()
    local path = _recv_bload(2, "H")
    movie.play(path, true)
end
register_action(movie_play)  -- 18


local function movie_record ()
    local path = _recv_bload(2, "H")
    local save_type = _unpack_load(1, "B")
    local author = _recv_bload(2, "H")
    movie.record(path, save_type, author)
end
register_action(movie_record)  -- 19


local function movie_stop ()
    movie.stop()
end
register_action(movie_stop)  -- 20


local function movie_rerecordcounting ()
    local counting = _unpack_load(1, "B")
    counting = string.byte(counting) > 0
    movie.rerecordcounting(counting)
end
register_action(movie_rerecordcounting)  -- 21


local function movie_rerecordcount ()
    local rerecordcount = movie.rerecordcount()
    _pack_load("L", rerecordcount)
end
register_action(movie_rerecordcount)  -- 22


return driver
