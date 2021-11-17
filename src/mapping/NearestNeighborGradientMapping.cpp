
#include "NearestNeighborGradientMapping.hpp"

#include <iostream>
#include <Eigen/Core>
#include <boost/container/flat_set.hpp>
#include <functional>
#include <memory>
#include "logging/LogMacros.hpp"
#include "mesh/Data.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/SharedPointer.hpp"
#include "mesh/Vertex.hpp"
#include "query/Index.hpp"
#include "utils/Event.hpp"
#include "utils/Statistics.hpp"
#include "utils/assertion.hpp"

namespace precice {
extern bool syncMode;

namespace mapping {

NearestNeighborGradientMapping::NearestNeighborGradientMapping(
    Constraint constraint,
    int        dimensions)
    : NearestNeighborBaseMapping(constraint, dimensions, true,"NearestNeighborGradientMapping", "nng" )
{
  if (hasConstraint(SCALEDCONSISTENT)) {
    setInputRequirement(Mapping::MeshRequirement::FULL);
    setOutputRequirement(Mapping::MeshRequirement::FULL);
  } else {
    setInputRequirement(Mapping::MeshRequirement::GRADIENT);
    setOutputRequirement(Mapping::MeshRequirement::VERTEX);
  }
}

double NearestNeighborGradientMapping::mapAt(int mapInputIndex, int vertex, const Eigen::VectorXd &inputValues, const Eigen::MatrixXd &gradientValues) {

  return inputValues(mapInputIndex) + _distancesMatched[vertex].transpose() * gradientValues.col(mapInputIndex); 
  
}

} // namespace mapping
} // namespace precice
