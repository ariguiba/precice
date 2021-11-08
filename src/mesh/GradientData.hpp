#pragma once

#include <Eigen/Core>
#include <stddef.h>
#include <string>

#include "logging/Logger.hpp"
#include "precice/types.hpp"

namespace precice {
namespace mesh {
class Mesh;
}
} // namespace precice

// ----------------------------------------------------------- CLASS DEFINITION

namespace precice {
namespace mesh {

/**
 * @brief Describes a set of gradient data values belonging to the vertices of a mesh.
 */
class GradientData {
public:

  /**
   * @brief Returns the number of created (and still existing) Gradient Data objects.
   *
   * Used to check if the number of data and corresponding gradient data is the same
   */
  static size_t getGradientDataCount();

  /**
   * @brief Sets the gradient data counter to zero.
   *
   * Used in test cases where multiple scenarios with gradient data are run.
   */
  static void resetGradientDataCount();

  /**
   * @brief Do not use this constructor! Only there for compatibility with std::map.
   */
  GradientData();

  /**
   * @brief Constructor.
   */

  GradientData(
      std::string name,
      DataID      id,
      int         meshDimension,
      int         dataDimension);

  /// Destructor, decrements gradient data count.
  ~GradientData();

  /// Returns a reference to the gradient data values.
  Eigen::MatrixXd &values();

  /// Returns a const reference to the gradient data values.
  const Eigen::MatrixXd &values() const;

  /// Returns the name of the gradient data set, as set in the config file.
  const std::string &getName() const;

  /// Returns the ID of the gradient data set (must be the same as corresponding Data set).
  DataID getID() const;

  /// Sets all values to zero
  void toZero();

  /// Returns the mesh dimension (i.e., number of rows) of one gradient data value .
  int getMeshDimensions() const;

  /// Returns the dimension (i.e., number of columns) of one gradient data value (must be the same as corresponding data set).
  int getDimensions() const;


private:
  logging::Logger _log{"mesh::GradientData"};

  /// Counter for existing Gradient Data objects.
  static size_t _gradientDataCount;

  Eigen::MatrixXd _values;

  /// Name of the gradient data set.
  std::string _name;

  /// ID of the gradient data set (should be the same as corresponding data set).
  DataID _id;

  /// Dimensionality of one mesh elements -> number of rows (only 1, 2, 3 allowed for 1D, 2D, 3D).
  int _meshDimensions;

  /// Dimensionality of one data value -> number of columns (scalar = 1, vector > 1).
  int _dimensions;

};

} // namespace mesh
} // namespace precice
