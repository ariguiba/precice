#pragma once

#include <string>
#include <vector>
#include "logging/Logger.hpp"
#include "mapping/NearestNeighborBaseMapping.hpp"

namespace precice {
namespace mapping {

/// Mapping using nearest neighboring vertices and their local gradient values.
class NearestNeighborGradientMapping : public NearestNeighborBaseMapping {
public:
  /**
   * @brief Constructor.
   *
   * @param[in] constraint Specifies mapping to be consistent or conservative.
   * @param[in] dimensions Dimensionality of the meshes
   */
  NearestNeighborGradientMapping(Constraint constraint, int dimensions);

  /// Destructor, empty.
  virtual ~NearestNeighborGradientMapping() {}
  
  virtual double mapAt(int mapInputIndex, int vertex, const Eigen::VectorXd &inputValues, const Eigen::MatrixXd &gradientValues) override ;
};  

} // namespace mapping
} // namespace precice
