#ifndef PRECICE_NO_MPI

#include <Eigen/Core>
#include <algorithm>
#include <deque>
#include <fstream>
#include <istream>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "action/RecorderAction.hpp"
#include "logging/LogMacros.hpp"
#include "math/constants.hpp"
#include "math/geometry.hpp"
#include "mesh/Data.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/SharedPointer.hpp"
#include "mesh/Utils.hpp"
#include "mesh/Vertex.hpp"
#include "precice/SolverInterface.hpp"
#include "precice/impl/MeshContext.hpp"
#include "precice/impl/Participant.hpp"
#include "precice/impl/SharedPointer.hpp"
#include "precice/impl/SolverInterfaceImpl.hpp"
#include "precice/types.hpp"
#include "testing/TestContext.hpp"
#include "testing/Testing.hpp"

using namespace precice;
using precice::testing::TestContext;

std::string pathToTests = testing::getPathToSources() + "/precice/tests/";

BOOST_AUTO_TEST_SUITE(PreciceTests)

BOOST_AUTO_TEST_SUITE(GradientTestRequirements)

BOOST_AUTO_TEST_CASE(NNG_Unidirectional_Parallel_A)
{
  PRECICE_TEST(1_rank);
  std::string filename = pathToTests + "nng-unidirectional-parallel.xml";

  SolverInterface interfaceA("A", filename, 0, 1);
  auto            meshIDA = interfaceA.getMeshID("MeshA");
  BOOST_TEST(interfaceA.isGradientRequired(meshIDA));
}

BOOST_AUTO_TEST_CASE(NNG_Unidirectional_Parallel_B)
{
  PRECICE_TEST(1_rank);
  std::string filename = pathToTests + "nng-unidirectional-parallel.xml";
  SolverInterface interfaceB("B", filename, 0, 1);
  auto            meshIDB = interfaceB.getMeshID("MeshB");
  BOOST_TEST(!interfaceB.isGradientRequired(meshIDB));
}

BOOST_AUTO_TEST_CASE(NNG_Unidirectional_Serial_A)
{
  PRECICE_TEST(1_rank);
  std::string filename = pathToTests + "nng-unidirectional-serial.xml";

  SolverInterface interfaceA("A", filename, 0, 1);
  auto            meshIDA = interfaceA.getMeshID("MeshA");
  BOOST_TEST(interfaceA.isGradientRequired(meshIDA));
}

BOOST_AUTO_TEST_CASE(NNG_Unidirectional_Serial_B)
{
  PRECICE_TEST(1_rank);
  std::string filename = pathToTests + "nng-unidirectional-serial.xml";
  SolverInterface interfaceB("B", filename, 0, 1);
  auto            meshIDB = interfaceB.getMeshID("MeshB");
  BOOST_TEST(!interfaceB.isGradientRequired(meshIDB));
}


// The second solver initializes the data of the first.
// A mapping is employed for the second solver, i.e., at the end of
// initializeData(), the mapping needs to be invoked.
// One direction : NNG ; The other : NN
BOOST_AUTO_TEST_CASE(NNG_Bidrectional_Serial_Explicit_Vector)
{
  PRECICE_TEST("SolverOne"_on(1_rank), "SolverTwo"_on(1_rank));

  using Eigen::Vector3d;
  using Eigen::Vector2d;

  SolverInterface cplInterface(context.name, pathToTests + "nng-explicit-serial-vector.xml", 0, 1);
  if (context.isNamed("SolverOne")) {
    int meshOneID = cplInterface.getMeshID("MeshOne");
    cplInterface.setMeshVertex(meshOneID, Vector3d(1.0, 2.0, 3.0).data());
    double maxDt      = cplInterface.initialize();
    int    dataAID    = cplInterface.getDataID("DataOne", meshOneID);
    int    dataBID    = cplInterface.getDataID("DataTwo", meshOneID);
    //double valueDataB = 0.0;
    //cplInterface.initializeData();
    //readScalar
    //BOOST_TEST(2.0 == valueDataB);

    Vector2d valueDataB;
    cplInterface.initializeData();
    cplInterface.readVectorData(dataBID, 0, valueDataB.data());
    Vector2d expected(-1.0, 0.0);
    BOOST_TEST(valueDataB == expected);

    while (cplInterface.isCouplingOngoing()) {
      Vector3d valueDataA(1.0, 1.0, 1.0);
      cplInterface.writeVectorData(dataAID, 0, valueDataA.data());

      maxDt = cplInterface.advance(maxDt);

      cplInterface.readVectorData(dataBID, 0, valueDataB.data());
      expected << -0.5, 0.5;
      BOOST_TEST(valueDataB == expected);

      //readScalar
      //BOOST_TEST(2.5 == valueDataB);
    }
    cplInterface.finalize();

  } else {
    BOOST_TEST(context.isNamed("SolverTwo"));
    int      meshTwoID = cplInterface.getMeshID("MeshTwo");
    Vector3d pos       = Vector3d::Zero();
    cplInterface.setMeshVertex(meshTwoID, pos.data());

    double maxDt   = cplInterface.initialize();
    int    dataAID = cplInterface.getDataID("DataOne", meshTwoID);
    int    dataBID = cplInterface.getDataID("DataTwo", meshTwoID);

    Vector2d valueDataB(2.0, 3.0);
    Vector2d gradient(1.0, 1.0);
    cplInterface.writeVectorData(dataBID, 0, valueDataB.data());
    cplInterface.writeGradientData(dataBID, 0, gradient.data(), gradient.data());


    //cplInterface.writeScalarData(dataBID, 0, 2.0);


    //tell preCICE that data has been written and call initializeData
    cplInterface.markActionFulfilled(precice::constants::actionWriteInitialData());
    cplInterface.initializeData();

    Vector3d valueDataA;
    cplInterface.readVectorData(dataAID, 0, valueDataA.data());
    Vector3d expected(1.0, 1.0, 1.0);
    BOOST_TEST(valueDataA == expected);

    while (cplInterface.isCouplingOngoing()) {
      //cplInterface.writeScalarData(dataBID, 0, 2.5);

      valueDataB << 2.5, 3.5;
      cplInterface.writeVectorData(dataBID, 0, valueDataB.data());
      cplInterface.writeGradientData(dataBID, 0, gradient.data(), gradient.data());

      maxDt = cplInterface.advance(maxDt);
      cplInterface.readVectorData(dataAID, 0, valueDataA.data());
      BOOST_TEST(valueDataA == expected);
    }
    cplInterface.finalize();
  }
}

// The second solver initializes the data of the first.
// A mapping is employed for the second solver, i.e., at the end of
// initializeData(), the mapping needs to be invoked.
// One direction : NNG ; The other : NN
BOOST_AUTO_TEST_CASE(NNG_Bidrectional_Serial_Explicit_Scalar)
{
  PRECICE_TEST("SolverOne"_on(1_rank), "SolverTwo"_on(1_rank));
  using Eigen::Vector3d;

  SolverInterface cplInterface(context.name, pathToTests + "nng-explicit-serial-scalar.xml", 0, 1);
  if (context.isNamed("SolverOne")) {
    int meshOneID = cplInterface.getMeshID("MeshOne");
    cplInterface.setMeshVertex(meshOneID, Vector3d(1.0, 2.0, 3.0).data());
    double maxDt      = cplInterface.initialize();
    int    dataAID    = cplInterface.getDataID("DataOne", meshOneID);
    int    dataBID    = cplInterface.getDataID("DataTwo", meshOneID);

    double valueDataB = 0.0;
    cplInterface.initializeData();
    cplInterface.readScalarData(dataBID, 0, valueDataB);
    BOOST_TEST(-4.0 == valueDataB);

    while (cplInterface.isCouplingOngoing()) {
      Vector3d valueDataA(1.0, 1.0, 1.0);
      cplInterface.writeVectorData(dataAID, 0, valueDataA.data());
      maxDt = cplInterface.advance(maxDt);

      cplInterface.readScalarData(dataBID, 0, valueDataB);
      BOOST_TEST(-3.5 == valueDataB);
    }
    cplInterface.finalize();

  } else {
    BOOST_TEST(context.isNamed("SolverTwo"));
    int      meshTwoID = cplInterface.getMeshID("MeshTwo");
    Vector3d pos       = Vector3d::Zero();
    cplInterface.setMeshVertex(meshTwoID, pos.data());

    double maxDt   = cplInterface.initialize();
    int    dataAID = cplInterface.getDataID("DataOne", meshTwoID);
    int    dataBID = cplInterface.getDataID("DataTwo", meshTwoID);

    double valueDataB = 2.0;
    cplInterface.writeScalarData(dataBID, 0, valueDataB);
    cplInterface.writeScalarGradientData(dataBID, 0, 1.0, 1.0, 1.0);


    //tell preCICE that data has been written and call initializeData
    cplInterface.markActionFulfilled(precice::constants::actionWriteInitialData());
    cplInterface.initializeData();

    Vector3d valueDataA;
    cplInterface.readVectorData(dataAID, 0, valueDataA.data());
    Vector3d expected(1.0, 1.0, 1.0);
    BOOST_TEST(valueDataA == expected);

    while (cplInterface.isCouplingOngoing()) {
      cplInterface.writeScalarData(dataBID, 0, 2.5);
      cplInterface.writeScalarGradientData(dataBID, 0, 1.0, 1.0, 1.0);

      maxDt = cplInterface.advance(maxDt);
      cplInterface.readVectorData(dataAID, 0, valueDataA.data());
      BOOST_TEST(valueDataA == expected);
    }
    cplInterface.finalize();
  }
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()


#endif // PRECICE_NO_MPI