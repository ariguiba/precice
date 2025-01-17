#ifndef PRECICE_NO_MPI

#include "testing/Testing.hpp"

#include <precice/SolverInterface.hpp>
#include <vector>

// Test case for a direct mesh access on one participant to a mesh defined
// by another participant (see above). In addition to the direct mesh access
// and data writing in one direction, an additional mapping (NN) is defined
// in the other direction.
BOOST_AUTO_TEST_SUITE(Integration)
BOOST_AUTO_TEST_SUITE(Parallel)
BOOST_AUTO_TEST_SUITE(DirectMeshAccess)
BOOST_AUTO_TEST_CASE(AccessReceivedMeshAndMapping)
{
  PRECICE_TEST("SolverOne"_on(2_ranks), "SolverTwo"_on(2_ranks));

  if (context.isNamed("SolverOne")) {
    // Set up Solverinterface
    precice::SolverInterface interface(context.name, context.config(), context.rank, context.size);
    BOOST_TEST(interface.getDimensions() == 2);
    constexpr int dim         = 2;
    const int     ownMeshID   = interface.getMeshID("MeshOne");
    const int     otherMeshID = interface.getMeshID("MeshTwo");
    const int     readDataID  = interface.getDataID("Forces", ownMeshID);
    const int     writeDataID = interface.getDataID("Velocities", otherMeshID);

    std::vector<double> positions = context.isMaster() ? std::vector<double>({0.0, 1.0, 0.0, 2.0, 0.0, 3.0}) : std::vector<double>({0.0, 4.0, 0.0, 5.0, 0.0, 6.0});

    std::vector<int> ownIDs(positions.size() / dim, 0);
    interface.setMeshVertices(ownMeshID, ownIDs.size(), positions.data(), ownIDs.data());

    std::array<double, dim * 2> boundingBox = context.isMaster() ? std::array<double, dim * 2>{0.0, 1.0, 0.0, 3.5} : std::array<double, dim * 2>{0.0, 1.0, 3.5, 5.0};
    // Define region of interest, where we could obtain direct write access
    interface.setMeshAccessRegion(otherMeshID, boundingBox.data());

    double dt = interface.initialize();
    // Get the size of the filtered mesh within the bounding box
    // (provided by the coupling participant)
    const int otherMeshSize = interface.getMeshVertexSize(otherMeshID);
    BOOST_TEST(otherMeshSize == 3);

    // Allocate a vector containing the vertices
    std::vector<double> solverTwoMesh(otherMeshSize * dim);
    std::vector<int>    otherIDs(otherMeshSize, 0);
    interface.getMeshVerticesAndIDs(otherMeshID, otherMeshSize, otherIDs.data(), solverTwoMesh.data());
    // Expected data = positions of the other participant's mesh
    const std::vector<double> expectedData = context.isMaster() ? std::vector<double>({0.0, 1.0, 0.0, 2.0, 0.0, 3.5}) : std::vector<double>({0.0, 3.5, 0.0, 4.0, 0.0, 5.0});
    BOOST_TEST(solverTwoMesh == expectedData);

    // Some dummy writeData
    std::vector<double> writeData;
    for (int i = 0; i < otherMeshSize; ++i)
      writeData.emplace_back(i + 5 + (10 * context.isMaster()));

    std::vector<double> readData(ownIDs.size(), 0);

    while (interface.isCouplingOngoing()) {
      // Write data
      interface.writeBlockScalarData(writeDataID, otherMeshSize,
                                     otherIDs.data(), writeData.data());
      dt = interface.advance(dt);
      interface.readBlockScalarData(readDataID, ownIDs.size(),
                                    ownIDs.data(), readData.data());

      // Expected data according to the writeData
      // Values are summed up
      std::vector<double> expectedData = context.isMaster() ? std::vector<double>({0, 1, 0}) : std::vector<double>({1, 2, 2});
      BOOST_TEST(precice::testing::equals(expectedData, readData));
    }

  } else {
    precice::SolverInterface interface(context.name, context.config(), context.rank, context.size);
    const int                dim = interface.getDimensions();
    BOOST_TEST(context.isNamed("SolverTwo"));
    std::vector<double> positions = context.isMaster() ? std::vector<double>({0.0, 1.0, 0.0, 2.0}) : std::vector<double>({0.0, 3.5, 0.0, 4.0, 0.0, 5.0});
    std::vector<int>    ids(positions.size() / dim, 0);

    // Query IDs
    const int meshID      = interface.getMeshID("MeshTwo");
    const int writeDataID = interface.getDataID("Forces", meshID);
    const int readDataID  = interface.getDataID("Velocities", meshID);

    // Define the mesh
    interface.setMeshVertices(meshID, ids.size(), positions.data(), ids.data());
    // Allocate data to read
    std::vector<double> readData(ids.size(), 0);
    std::vector<double> writeData;
    for (unsigned int i = 0; i < ids.size(); ++i)
      writeData.emplace_back(i);

    // Initialize
    double dt = interface.initialize();
    while (interface.isCouplingOngoing()) {

      interface.writeBlockScalarData(writeDataID, ids.size(),
                                     ids.data(), writeData.data());
      dt = interface.advance(dt);
      interface.readBlockScalarData(readDataID, ids.size(),
                                    ids.data(), readData.data());
      // Expected data according to the writeData
      // Values are summed up
      std::vector<double> expectedData = context.isMaster() ? std::vector<double>({15, 16}) : std::vector<double>({22, 6, 7});
      BOOST_TEST(precice::testing::equals(expectedData, readData));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END() // Integration
BOOST_AUTO_TEST_SUITE_END() // Parallel
BOOST_AUTO_TEST_SUITE_END() // DirectMeshAccess

#endif // PRECICE_NO_MPI
