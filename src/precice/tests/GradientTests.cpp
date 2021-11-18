#ifndef PRECICE_NO_MPI //what is this ? 

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

BOOST_AUTO_TEST_SUITE(PreciceTests)

BOOST_AUTO_TEST_SUITE(GradientTestRequirements)

BOOST_AUTO_TEST_CASE(NNG_A)
{
    PRECICE_TEST(1_rank);
    std::string pathToTests = testing::getPathToSources() + "/precice/tests/";
    std::string     filename = pathToTests + "meshrequirements-nng.xml";
    SolverInterface interface("A", filename, 0, 1);
    auto            meshID = interface.getMeshID("MeshA");
    BOOST_TEST(interface.isGradientRequired(meshID)); // this should be wrong (no mapping set)
}

BOOST_AUTO_TEST_CASE(NNG_B)
{
  PRECICE_TEST(1_rank);
  std::string pathToTests = testing::getPathToSources() + "/precice/tests/";
  std::string     filename = pathToTests + "meshrequirements-nng.xml";
  SolverInterface interface("B", filename, 0, 1);
  auto            meshID = interface.getMeshID("MeshB");
  BOOST_TEST(!interface.isGradientRequired(meshID));
}

BOOST_AUTO_TEST_CASE(Full)
{
    PRECICE_TEST("SolverOne"_on(1_rank), "SolverTwo"_on(1_rank));
    std::string pathToTests = testing::getPathToSources() + "/precice/tests/";
    std::string config = pathToTests + "meshrequirements-nng-full.xml";

    SolverInterface interface(context.name, config, context.rank, context.size);

  if (context.isNamed("SolverOne")) {
    auto   meshid   = interface.getMeshID("MeshOne");
    double coords[] = {0.1, 1.2, 2.3};
    auto   vertexid = interface.setMeshVertex(meshid, coords);

    auto   dataid = interface.getDataID("DataOne", meshid);
    double data[] = {3.4, 4.5, 5.6};
    double gradientDataX[] = {1.0, 1.0, 1.0};
    double gradientDataY[] = {1.0, 1.0, 1.0};
    double gradientDataZ[] = {1.0, 1.0, 1.0};
    interface.writeVectorData(dataid, vertexid, data);
    interface.writeGradientData(dataid, vertexid, gradientDataX, gradientDataY, gradientDataZ);

  } else {
    auto   meshid   = interface.getMeshID("MeshTwo");
    double coords[] = {0.12, 1.21, 2.2};
    auto   vertexid = interface.setMeshVertex(meshid, coords);

    auto dataid = interface.getDataID("DataTwo", meshid);
    interface.writeScalarData(dataid, vertexid, 7.8);
  }
  interface.initialize();
  BOOST_TEST(interface.isCouplingOngoing());
  interface.finalize();
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()


#endif // PRECICE_NO_MPI