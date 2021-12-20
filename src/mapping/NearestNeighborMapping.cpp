#include "NearestNeighborMapping.hpp"

#include <Eigen/Core>
#include <boost/container/flat_set.hpp>
#include <functional>
#include "logging/LogMacros.hpp"
#include "utils/Event.hpp"
#include "utils/assertion.hpp"
#include "utils/EigenHelperFunctions.hpp"
#include "utils/EventUtils.hpp"

namespace precice {
extern bool syncMode;

namespace mapping {

NearestNeighborMapping::NearestNeighborMapping(
    Constraint constraint,
    int        dimensions)
    : NearestNeighborBaseMapping(constraint, dimensions, false, "NearestNeighborMapping", "nn" )
{
  if (hasConstraint(SCALEDCONSISTENT)) {
    setInputRequirement(Mapping::MeshRequirement::FULL);
    setOutputRequirement(Mapping::MeshRequirement::FULL);
  } else {
    setInputRequirement(Mapping::MeshRequirement::VERTEX);
    setOutputRequirement(Mapping::MeshRequirement::VERTEX);
  }
}

void NearestNeighborMapping::map(
    int inputDataID,
    int outputDataID)
{
  PRECICE_TRACE(inputDataID, outputDataID);

  precice::utils::Event e("map." + MAPPING_NAME_SHORT + ".mapData.From" + input()->getName() + "To" + output()->getName(), precice::syncMode);
  int valueDimensions = input()->data(inputDataID)->getDimensions(); // Data dimensions (bei scalar = 1, bei vectors > 1)

  const Eigen::VectorXd &inputValues  = input()->data(inputDataID)->values();
  Eigen::VectorXd &      outputValues = output()->data(outputDataID)->values();


  //assign(outputValues) = 0.0;

  PRECICE_ASSERT(valueDimensions == output()->data(outputDataID)->getDimensions(),
                 valueDimensions, output()->data(outputDataID)->getDimensions());
  PRECICE_ASSERT(inputValues.size() / valueDimensions == (int) input()->vertices().size(),
                 inputValues.size(), valueDimensions, input()->vertices().size());
  PRECICE_ASSERT(outputValues.size() / valueDimensions == (int) output()->vertices().size(),
                 outputValues.size(), valueDimensions, output()->vertices().size());


  if (hasConstraint(CONSERVATIVE)) {
    PRECICE_DEBUG("Map conservative");
    size_t const inSize = input()->vertices().size();

    for (size_t i = 0; i < inSize; i++) {
      int const outputIndex = _vertexIndices[i] * valueDimensions;

      for (int dim = 0; dim < valueDimensions; dim++) {

        int mapOutputIndex = outputIndex + dim;
        int mapInputIndex = (i * valueDimensions) + dim;

        outputValues(mapOutputIndex) += inputValues(mapInputIndex);

      }
    }
  } else {
    PRECICE_DEBUG((hasConstraint(CONSISTENT) ? "Map consistent" : "Map scaled-consistent"));
    size_t const outSize = output()->vertices().size();

    for (size_t i = 0; i < outSize; i++) {
      int inputIndex = _vertexIndices[i] * valueDimensions;

      for (int dim = 0; dim < valueDimensions; dim++) {

        int mapOutputIndex = (i * valueDimensions) + dim;
        int mapInputIndex =  inputIndex + dim;

        outputValues(mapOutputIndex) = inputValues(mapInputIndex);
      }
    }
    if (hasConstraint(SCALEDCONSISTENT)) {
      scaleConsistentMapping(inputDataID, outputDataID);
    }

    PRECICE_DEBUG("Mapped values = {}", utils::previewRange(3, outputValues));
  }
}

} // namespace mapping
} // namespace precice
