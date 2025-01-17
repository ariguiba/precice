#pragma once

#include "precice/impl/DataContext.hpp"
#include "precice/impl/MappingContext.hpp"

namespace precice {
namespace testing {
/*
 * @brief A fixture that is used to access private functions of the DataContext class.
 *
 * The fixture can be used to call private functions for individual testing.
 */
class DataContextFixture {
public:
  impl::MappingContext mappingContext(precice::impl::DataContext &dataContext);

  int getProvidedDataID(precice::impl::DataContext &dataContext);

  int getFromDataID(precice::impl::DataContext &dataContext);

  int getToDataID(precice::impl::DataContext &dataContext);

  bool hasMapping(precice::impl::DataContext &dataContext);

  bool hasReadMapping(precice::impl::DataContext &dataContext);

  bool hasWriteMapping(precice::impl::DataContext &dataContext);
};

} // namespace testing
} // namespace precice
