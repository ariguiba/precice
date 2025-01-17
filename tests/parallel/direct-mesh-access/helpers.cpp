#ifndef PRECICE_NO_MPI

#include "helpers.hpp"

#include "precice/SolverInterface.hpp"
#include "testing/Testing.hpp"

// StartIndex is here the first index to be used for writing on the slave rank
void runTestAccessReceivedMesh(const TestContext &       context,
                               const std::vector<double> boundingBoxSlave,
                               const std::vector<double> writeDataSlave,
                               const std::vector<double> expectedPositionSlave,
                               const std::vector<double> expectedReadDataSlave,
                               const int                 startIndex)
{
  if (context.isNamed("SolverOne")) {
    // Defines the bounding box and writes data to the received mesh
    precice::SolverInterface interface(context.name, context.config(), context.rank, context.size);
    const int                otherMeshID = interface.getMeshID("MeshTwo");
    const int                dataID      = interface.getDataID("Velocities", otherMeshID);
    const int                dim         = interface.getDimensions();

    std::vector<double> boundingBox = context.isMaster() ? std::vector<double>({0.0, 1.0, 0.0, 3.5}) : boundingBoxSlave;
    // Set bounding box
    interface.setMeshAccessRegion(otherMeshID, boundingBox.data());
    // Initialize the solverinterface
    double dt = interface.initialize();

    // Get relevant size, allocate data structures and retrieve coordinates
    const int meshSize = interface.getMeshVertexSize(dataID);

    // According to the bounding boxes and vertices: the master rank receives 3 vertices, the slave rank 2
    const bool expectedSize = (context.isMaster() && meshSize == 3) ||
                              (!context.isMaster() && meshSize == static_cast<int>(expectedPositionSlave.size() / dim));
    BOOST_TEST(expectedSize);

    // Allocate memory
    std::vector<int>    ids(meshSize);
    std::vector<double> coordinates(meshSize * dim);
    interface.getMeshVerticesAndIDs(otherMeshID, meshSize, ids.data(), coordinates.data());

    // Check the received vertex coordinates
    std::vector<double> expectedPositions = context.isMaster() ? std::vector<double>({0.0, 1.0, 0.0, 2.0, 0.0, 3.0}) : expectedPositionSlave;
    BOOST_TEST(testing::equals(expectedPositions, coordinates));

    // Check the received vertex IDs (IDs are local?!)
    std::vector<int> expectedIDs;
    for (int i = 0; i < meshSize; ++i)
      expectedIDs.emplace_back(i);
    BOOST_TEST(expectedIDs == ids);

    // Create some unique writeData in order to check it in the other participant
    std::vector<double> writeData = context.isMaster() ? std::vector<double>({1, 2, 3}) : writeDataSlave;

    while (interface.isCouplingOngoing()) {
      // Write data
      if (context.isMaster()) {
        interface.writeBlockScalarData(dataID, meshSize,
                                       ids.data(), writeData.data());
      } else {
        interface.writeBlockScalarData(dataID, meshSize - startIndex,
                                       &ids[startIndex], writeData.data());
      }

      dt = interface.advance(dt);
    }
  } else {
    // Defines the mesh and reads data
    BOOST_REQUIRE(context.isNamed("SolverTwo"));
    precice::SolverInterface interface(context.name, context.config(), context.rank, context.size);
    BOOST_TEST(interface.getDimensions() == 2);

    // Get IDs
    const int meshID = interface.getMeshID("MeshTwo");
    const int dataID = interface.getDataID("Velocities", meshID);
    const int dim    = interface.getDimensions();
    // Define the interface
    std::vector<double> positions = context.isMaster() ? std::vector<double>({0.0, 1.0, 0.0, 2.0}) : std::vector<double>({0.0, 3.0, 0.0, 4.0, 0.0, 5.0});

    const int        size = positions.size() / dim;
    std::vector<int> ids(size);

    interface.setMeshVertices(meshID, size, positions.data(), ids.data());

    {
      // Check, if we can use the 'getMeshVerticesAndIDs' function on provided meshes as well,
      // though the actual purpose is of course using it on received meshes
      const int ownMeshSize = interface.getMeshVertexSize(meshID);
      BOOST_TEST(ownMeshSize == size);
      std::vector<int>    ownIDs(ownMeshSize);
      std::vector<double> ownCoordinates(ownMeshSize * dim);
      interface.getMeshVerticesAndIDs(meshID, ownMeshSize, ownIDs.data(), ownCoordinates.data());
      BOOST_TEST(ownIDs == ids);
      BOOST_TEST(testing::equals(positions, ownCoordinates));
    }

    // Initialize the solverinterface
    double dt = interface.initialize();

    // Start the time loop
    std::vector<double> readData(size);
    while (interface.isCouplingOngoing()) {

      dt = interface.advance(dt);
      interface.readBlockScalarData(dataID, size,
                                    ids.data(), readData.data());

      // Check the received data
      const std::vector<double> expectedReadData = context.isMaster() ? std::vector<double>({1, 2}) : expectedReadDataSlave;
      BOOST_TEST(expectedReadData == readData);
    }
  }
}

#endif
