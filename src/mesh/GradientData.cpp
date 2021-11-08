#include "GradientData.hpp"
#include <algorithm>
#include <utility>

#include "precice/types.hpp"
#include "utils/assertion.hpp"

namespace precice {
namespace mesh {

size_t GradientData::_gradientDataCount = 0;

GradientData::GradientData()
    : _name(""),
      _id(-1),
      _meshDimensions(0),
      _dimensions(0)
      
{
  PRECICE_ASSERT(false);
}

GradientData::GradientData(
    std::string name,
    DataID      id,
    int         meshDimensions,
    int         dimensions)
    : _values(),
      _name(std::move(name)),
      _id(id),
      _meshDimensions(meshDimensions),
      _dimensions(dimensions)
      
{
  PRECICE_ASSERT(meshDimensions > 0, meshDimensions);
  PRECICE_ASSERT(dimensions > 0, dimensions);

  _gradientDataCount++;
}

GradientData::~GradientData()
{
  _gradientDataCount--;
}

Eigen::MatrixXd &GradientData::values()
{
  return _values;
}

const Eigen::MatrixXd &GradientData::values() const
{
  return _values;
}

const std::string &GradientData::getName() const
{
  return _name;
}

DataID GradientData::getID() const
{
  return _id;
}

void GradientData::toZero()
{
  auto begin = _values.data(); 
  auto end   = begin + _values.size();
  std::fill(begin, end, 0.0);
}

int GradientData::getDimensions() const
{
  return _dimensions;
}

int GradientData::getMeshDimensions() const
{
  return _meshDimensions;
}

size_t GradientData::getGradientDataCount()
{
  return _gradientDataCount;
}

void GradientData::resetGradientDataCount()
{
  _gradientDataCount = 0;
}

} // namespace mesh
} // namespace precice
