include("../src/iTEBD.jl")
import .iTEBD: fixed_points, right_cannonical

const aklt = begin
    aklt_tensor = zeros(4,3,4)
    aklt_tensor[1,1,2] = +sqrt(2/3)
    aklt_tensor[1,2,1] = -sqrt(1/3)
    aklt_tensor[2,2,2] = +sqrt(1/3)
    aklt_tensor[2,3,1] = -sqrt(2/3)
    aklt_tensor[3,1,4] = +sqrt(2/3)
    aklt_tensor[3,2,3] = -sqrt(1/3)
    aklt_tensor[4,2,4] = +sqrt(1/3)
    aklt_tensor[4,3,3] = -sqrt(2/3)
    aklt_tensor
end

res = right_cannonical(aklt)