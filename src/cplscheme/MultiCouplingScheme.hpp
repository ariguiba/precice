#pragma once

#include <map>
#include <string>
#include <vector>
#include "BaseCouplingScheme.hpp"
#include "cplscheme/Constants.hpp"
#include "logging/Logger.hpp"
#include "m2n/SharedPointer.hpp"
#include "mesh/SharedPointer.hpp"
#include "utils/Helpers.hpp"

namespace precice {
namespace cplscheme {
class CouplingData;
struct ExchangeData;

/**
 * @brief A coupling scheme with multiple participants.
 *
 * ! General description
 * A MultiCouplingScheme couples multiple participants in a fully implicit fashion.
 * It is a specialization of BaseCouplingScheme.
 *
 */
class MultiCouplingScheme : public BaseCouplingScheme {
public:
  /**
 * @brief Constructor.
 *
 * @param[in] maxTime Simulation time limit, or UNDEFINED_TIME.
 * @param[in] maxTimeWindows Simulation time windows limit, or UNDEFINED_TIMEWINDOWS.
 * @param[in] timeWindowSize Simulation time window size.
 * @param[in] validDigits valid digits for computation of the remainder of a time window
 * @param[in] localParticipant Name of participant using this coupling scheme.
 * @param[in] m2ns M2N communications to all other participants of coupling scheme.
 * @param[in] dtMethod Method used for determining the time window size, see https://www.precice.org/couple-your-code-timestep-sizes.html
 * @param[in] maxIterations maximum number of coupling sub-iterations allowed.
 * @param[in] extrapolationOrder order used for extrapolation
 */
  MultiCouplingScheme(
      double                             maxTime,
      int                                maxTimeWindows,
      double                             timeWindowSize,
      int                                validDigits,
      const std::string &                localParticipant,
      std::map<std::string, m2n::PtrM2N> m2ns,
      constants::TimesteppingMethod      dtMethod,
      const std::string &                controller,
      int                                maxIterations,
      int                                extrapolationOrder);

  /// Adds data to be sent on data exchange and possibly be modified during coupling iterations.
  void addDataToSend(
      const mesh::PtrData &data,
      mesh::PtrMesh        mesh,
      bool                 initialize,
      const std::string &  to);

  /// Adds data to be received on data exchange.
  void addDataToReceive(
      const mesh::PtrData &data,
      mesh::PtrMesh        mesh,
      bool                 initialize,
      const std::string &  from);

  /// returns list of all coupling partners
  std::vector<std::string> getCouplingPartners() const override final;

  /**
   * @returns true, if coupling scheme has any sendData
   */
  bool hasAnySendData() override final
  {
    return std::any_of(_sendDataVector.cbegin(), _sendDataVector.cend(), [](const auto &sendExchange) { return not sendExchange.second.empty(); });
  }

private:
  /**
   * @brief A vector of m2ns. A m2n is a communication device to the other coupling participant.
   */
  std::map<std::string, m2n::PtrM2N> _m2ns;

  /**
   * @brief A vector of all data to be received.
   */
  std::map<std::string, DataMap> _receiveDataVector;

  /**
   * @brief A vector of all data to be sent.
   */
  std::map<std::string, DataMap> _sendDataVector;

  logging::Logger _log{"cplscheme::MultiCouplingScheme"};

  /**
   * @brief BiCouplingScheme has _sendData and _receiveData
   * @returns DataMap with all data
   */
  const DataMap getAllData() override
  {
    DataMap allData;
    // @todo user C++17 std::map::merge
    for (auto &sendData : _sendDataVector) {
      allData.insert(sendData.second.begin(), sendData.second.end());
    }
    for (auto &receiveData : _receiveDataVector) {
      allData.insert(receiveData.second.begin(), receiveData.second.end());
    }
    return allData;
  }

  /**
   * @brief Exchanges all data between the participants of the MultiCouplingScheme and applies acceleration.
   * @returns true, if iteration converged
   */
  bool exchangeDataAndAccelerate() override;

  /**
   * @brief MultiCouplingScheme applies acceleration to all CouplingData
   * @returns DataMap being accelerated
   */
  const DataMap getAccelerationData() override
  {
    return getAllData();
  }

  /**
   * @brief Initialization of MultiCouplingScheme is similar to ParallelCouplingScheme. We only have to iterate over all pieces of data in _sendDataVector and _receiveDataVector.
   */
  void initializeImplementation() override;

  /**
   * @brief Exchanges data, if it has to be initialized.
   */
  void exchangeInitialData() override;

  /// name of the controller participant
  std::string _controller;

  /// if this is the controller or not
  bool _isController;
};

} // namespace cplscheme
} // namespace precice
