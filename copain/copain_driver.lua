local socket = require "posix.sys.socket"
local unistd = require "posix.unistd"

local to_uint16 = (require "utils").to_uint16


local driver = { }


function driver.start ()
    local socket_path = assert(os.getenv("COPAIN_SOCKET_SERVER_FILE_PATH"))
    driver.socket_fd = assert(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0))
    socket_fd = driver.socket_fd
    assert(socket.connect(socket_fd, {family = socket.AF_UNIX, path = socket_path}))
end


function driver.stop ()
    assert(socket.shutdown(socket_fd, socket.SHUT_RDWR))
    assert(unistd.close(socket_fd))
    socket_fd = false
    driver.socket_fd = socket_fd
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


-- emu.* namespace

local function emu_poweron ()
    emu.poweron()
end
register_action(emu_poweron)  -- 1


local SPEEDMODE_STR = {"normal", "nothrottle", "turbo", "maximum"}
local function emu_speedmode ()
    local speedmode_code = assert(socket.recv(socket_fd, 1))
    speedmode_code = string.byte(speedmode_code)
    emu.speedmode(SPEEDMODE_STR[speedmode_code])
end
register_action(emu_speedmode)  -- 2


local function emu_frameadvance ()
    emu.frameadvance ()
end
register_action(emu_frameadvance)  -- 3


local function emu_nframeadvance ()
    local n = assert(socket.recv(socket_fd, 1))
    n = string.byte(n)
    for step=1,n,1 do
        emu.frameadvance ()
    end
end
register_action(emu_nframeadvance)  -- 4


local function emu_setrenderplanes ()
    local args = assert(socket.recv(socket_fd, 2))
    local sprites, background = string.byte(args, 1, 2)
    sprites = sprites == 1
    background = background == 1
    emu.setrenderplanes(sprites, background)
end
register_action(emu_setrenderplanes)  -- 5


local function emu_loadrom ()
    local path = assert(socket.recv(socket_fd, 2))
    path = to_uint16(path)
    path = assert(socket.recv(socket_fd, path))
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
    local hashtype = assert(socket.recv(socket_fd, 1))
    hashtype = HASHTYPE_STR[string.byte(hashtype)]
    socket.send(socket_fd, rom.gethash (hashtype))
end
register_action(rom_gethash)  -- 8


-- memory.* namespace

local function memory_readbyterange ()
    local address = assert(socket.recv(socket_fd, 2))
    local length = assert(socket.recv(socket_fd, 2))
    address = to_uint16(address)
    length = to_uint16(length)
    local memory = memory.readbyterange(address, length)
    socket.send(socket_fd, memory)
end
register_action(memory_readbyterange)  -- 9


-- ppu.* namespace

local function ppu_readbyterange ()
    local address = assert(socket.recv(socket_fd, 2))
    local length = assert(socket.recv(socket_fd, 2))
    address = to_uint16(address)
    length = to_uint16(length)
    local memory = ppu.readbyterange(address, length)
    socket.send(socket_fd, memory)
end
register_action(ppu_readbyterange)  -- 10


-- joypad.* namespace

local function joypad_set()
    local args = assert(socket.recv(socket_fd, 9))
    local player, up, down, left, right, A, B, start, selec = string.byte(args, 1, 9)
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
    local n = assert(socket.recv(socket_fd, 1))
    local savestate_id = assert(socket.recv(socket_fd, 16))
    n = string.byte(n)
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


-- misc

local function get_runner_id()
    local runner_id = assert(os.getenv("COPAIN_RUN_ID"))
    if string.len(runner_id) == 1 then runner_id = runner_id .. "\0" end
    socket.send(socket_fd, runner_id)
end
register_action(get_runner_id)  -- 18


return driver
