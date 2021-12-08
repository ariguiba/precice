
#include "NearestNeighborBaseMapping.hpp"

#include <iostream>
#include <Eigen/Core>
#include <boost/container/flat_set.hpp>
#include <functional>
#include <memory>
#include "logging/LogMacros.hpp"
#include "mapping/Mapping.hpp"
#include "mesh/Data.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/SharedPointer.hpp"
#include "mesh/Vertex.hpp"
#include "query/Index.hpp"
#include "utils/Event.hpp"
#include "utils/Statistics.hpp"
#include "utils/assertion.hpp"
#include "utils/EigenHelperFunctions.hpp"
#include "utils/EigenIO.hpp"
#include "utils/EventUtils.hpp"
#include "utils/Helpers.hpp"
#include "utils/MasterSlave.hpp"
#include "utils/Parallel.hpp"
#include "utils/Petsc.hpp"
#include "utils/PointerVector.hpp"
#include "utils/algorithm.hpp"


namespace precice {
extern bool syncMode;

namespace mapping {

NearestNeighborBaseMapping::NearestNeighborBaseMapping(
  Constraint constraint, 
  int dimensions, 
  bool hasGradient,
  std::string mappingName,
  std::string mappingNameShort)
  : Mapping(constraint, dimensions),
    _hasGradient(hasGradient), 
    MAPPING_NAME(mappingName),
    MAPPING_NAME_SHORT(mappingNameShort) 
{
}

bool NearestNeighborBaseMapping::hasGradient() {
  return _hasGradient;
}

void NearestNeighborBaseMapping::computeMapping()
{
  PRECICE_TRACE(input()->vertices().size());

  PRECICE_ASSERT(input().get() != nullptr);
  PRECICE_ASSERT(output().get() != nullptr);

  const std::string     baseEvent = "map." + MAPPING_NAME_SHORT + ".computeMapping.From" + input()->getName() + "To" + output()->getName();
  precice::utils::Event e(baseEvent, precice::syncMode);

  // Setup Direction of Mapping
  mesh::PtrMesh origins, searchSpace;
  if (hasConstraint(CONSERVATIVE)) {
    PRECICE_DEBUG("Compute conservative mapping");
    origins     = input();
    searchSpace = output();
  } else {
    PRECICE_DEBUG((hasConstraint(CONSISTENT) ? "Compute consistent mapping" : "Compute scaled-consistent mapping"));
    origins     = output();
    searchSpace = input();
  }

  precice::utils::Event e2(baseEvent + ".getIndexOnVertices", precice::syncMode);
  query::Index          indexTree(searchSpace);
  e2.stop();

  const size_t verticesSize   = origins->vertices().size();
  const auto & sourceVertices = origins->vertices();

  _vertexIndices.resize(verticesSize); 

  if (hasGradient())
    _distancesMatched.resize(verticesSize); 


  utils::statistics::DistanceAccumulator distanceStatistics; 

  for (size_t i = 0; i < verticesSize; ++i) {
    auto matchedVertex = indexTree.getClosestVertex(sourceVertices[i].getCoords());
    

    // Match the difference vector between the source vector and the matched one (relevant for gradient) 
    if(hasGradient()) {
      auto matchedVertexCoords = searchSpace.get()->vertices()[matchedVertex.index].getCoords();
      _distancesMatched[i] = matchedVertexCoords - sourceVertices[i].getCoords();

      if (hasConstraint (CONSERVATIVE)) _distancesMatched[i] *= -1;
    }

    _vertexIndices[i]  = matchedVertex.index;
    distanceStatistics(matchedVertex.distance);
  }

  if (distanceStatistics.empty()) {
    PRECICE_INFO("Mapping distance not available due to empty partition.");
  } else {
    PRECICE_INFO("Mapping distance {}", distanceStatistics);
  }

  _hasComputedMapping = true;
}

bool NearestNeighborBaseMapping::hasComputedMapping() const
{
  PRECICE_TRACE(_hasComputedMapping);
  return _hasComputedMapping;
}

void NearestNeighborBaseMapping::clear()
{
  PRECICE_TRACE();
  _vertexIndices.clear();
  _hasComputedMapping = false;

  if (hasGradient())
    _distancesMatched.clear();

  if (getConstraint() == CONSISTENT) {
    query::clearCache(input()->getID());
  } else {
    query::clearCache(output()->getID());
  }
}


void NearestNeighborBaseMapping::map(
    int inputDataID,
    int outputDataID)
{
  PRECICE_TRACE(inputDataID, outputDataID);

  precice::utils::Event e("map." + MAPPING_NAME_SHORT + ".mapData.From" + input()->getName() + "To" + output()->getName(), precice::syncMode);

  int meshDimensions = input()->getDimensions(); // Vector dimensions 
  int valueDimensions = input()->data(inputDataID)->getDimensions(); // Data dimensions (bei scalar = 1, bei vectors > 1)

  const Eigen::VectorXd &inputValues  = input()->data(inputDataID)->values();
  Eigen::VectorXd &      outputValues = output()->data(outputDataID)->values();

  // If it's NearestNeighborMapping check if gradient data is available
  
  if (hasGradient()){
    /// Check if input has gradient data, else send Error
    if (!input()->vertices().empty() && !input()->data(inputDataID)->hasGradient()){
        PRECICE_ERROR("Mesh \"{}\" does not contain gradient data. ",
                    input()->getName());
    }
  }


  const Eigen::MatrixXd &gradientValues = input()->data(inputDataID)->gradientValues(); 
  
  
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

        outputValues(mapOutputIndex) += mapAt(mapInputIndex, i, inputValues, gradientValues);

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

        outputValues(mapOutputIndex) = mapAt(mapInputIndex, i, inputValues, gradientValues);
      }
    }
    if (hasConstraint(SCALEDCONSISTENT)) {
      scaleConsistentMapping(inputDataID, outputDataID);
    }

    PRECICE_DEBUG("Mapped values (w/o gradient) = {}", utils::previewRange(3, outputValues));
  }
}

void NearestNeighborBaseMapping::tagMeshFirstRound()
{
  PRECICE_TRACE();
  precice::utils::Event e("map."+ MAPPING_NAME_SHORT + ".tagMeshFirstRound.From" + input()->getName() + "To" + output()->getName(), precice::syncMode);

  computeMapping();

  // Lookup table of all indices used in the mapping
  const boost::container::flat_set<int> indexSet(_vertexIndices.begin(), _vertexIndices.end());

  // Get the source mesh depending on the constraint
  const mesh::PtrMesh &source = hasConstraint(CONSERVATIVE) ? output() : input();

  // Tag all vertices used in the mapping
  for (mesh::Vertex &v : source->vertices()) {
    if (indexSet.count(v.getID()) != 0) {
      v.tag();
    }
  }

  clear();
}

void NearestNeighborBaseMapping::tagMeshSecondRound()
{
  PRECICE_TRACE();
  // for NNG mapping no operation needed here
}

} // namespace mapping
} // namespace precice
