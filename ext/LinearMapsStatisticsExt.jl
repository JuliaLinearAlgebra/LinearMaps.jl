module LinearMapsStatisticsExt

import Statistics: mean

using LinearMaps
using LinearMaps: LinearMapTupleOrVector, LinearCombination

mean(f::F, maps::LinearMapTupleOrVector) where {F} = sum(f, maps) / length(maps)
mean(maps::LinearMapTupleOrVector) = mean(identity, maps)
mean(A::LinearCombination) = mean(A.maps)

end # module ChainRulesCore
