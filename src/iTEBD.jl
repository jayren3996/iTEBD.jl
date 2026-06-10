module iTEBD

import InfiniteTEBD

const _SKIP_REEXPORT = (:eval, :include, :InfiniteTEBD)

_reexports_name(name::Symbol) = name ∉ _SKIP_REEXPORT && !startswith(String(name), "#")

for name in names(InfiniteTEBD; all=true, imported=false)
    _reexports_name(name) || continue
    @eval const $(name) = InfiniteTEBD.$(name)
end

for name in names(InfiniteTEBD; all=false, imported=false)
    @eval export $(name)
end

end
