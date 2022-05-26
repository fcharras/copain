local unpack = (require "struct").unpack

local driver = require "copain_driver"


local function main ()
    driver.start ()

    while true do
        local action_code = driver.recv_action_code ()
        if string.len(action_code) == 0 then break end
        action_code = unpack("H", action_code)
        driver.actions_registry[action_code] ()
    end
end

--[[
-- TODO: after upgrading to lua > 5.1, use this instead to ensure correct cleaning
local outcome = { pcall(main) }
if not outcome[1] then emu.print(outcome[2]) end
--]]
main()
driver.stop ()
emu.exit()
