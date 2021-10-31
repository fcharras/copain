utils = {}

function utils.to_uint16 (str)
    assert(string.len(str) == 2)
    local a, b = string.byte(str, 0, 2)
    return (b * (2 ^ 8) + a)
end

return utils
