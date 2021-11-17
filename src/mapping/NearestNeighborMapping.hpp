#pragma once

#include <string>
#include <vector>
#include "logging/Logger.hpp"
#include "mapping/NearestNeighborBaseMapping.hpp"

namespace precice {
namespace mapping {

/// Mapping using nearest neighboring vertices.
class NearestNeighborMapping : public NearestNeighborBaseMapping {
public:
  /**
   * @brief Constructor.
   *
   * @param[in] constraint Specifies mapping to be consistent or conservative.
   * @param[in] dimensions Dimensionality of the meshes
   */
  NearestNeighborMapping(Constraint constraint, int dimensions);

  /// Destructor, empty.
  virtual ~NearestNeighborMapping() {}

  virtual double mapAt(int mapInputIndex, int vertex, const Eigen::VectorXd &inputValues, const Eigen::MatrixXd &gradientValues) override ;
}; 

} // namespace mapping
} // namespace precice
