#include "../impl/WatchPoint.hpp"
#include "geometry/Cuboid.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/SharedPointer.hpp"
#include "mesh/Vertex.hpp"
#include "testing/Testing.hpp"

using namespace precice;

BOOST_AUTO_TEST_SUITE(Precice)

BOOST_AUTO_TEST_CASE(WatchPoint)
{
  using namespace mesh;
  int dim = 2;
  using Eigen::VectorXd;
  // Setup geometry
  std::string name("rectangle");
  bool flipNormals = false;
  PtrMesh mesh(new Mesh(name, dim, flipNormals));
  VectorXd offset = VectorXd::Zero(dim);
  double discretizationWidth = 0.5;
  VectorXd sidelengths = VectorXd::Constant(dim, 1.0);
  geometry::Cuboid rectangleGeometry(offset, discretizationWidth, sidelengths);
  PtrData doubleData = mesh->createData("DoubleData", 1);
  PtrData vectorData = mesh->createData("VectorData", dim);
  auto &doubleValues = doubleData->values();
  auto &vectorValues = vectorData->values();
  rectangleGeometry.create(*mesh);

  // Create watchpoints
  VectorXd pointToWatch0 = VectorXd::Constant(dim, 1.0);
  std::string filename0("tests-WatchPointTest-output0.txt");
  impl::WatchPoint watchpoint0(pointToWatch0, mesh, filename0);
  Eigen::Vector2d pointToWatch1(1.0, 0.5);
  std::string filename1("tests-WatchPointTest-output1.txt");
  impl::WatchPoint watchpoint1(pointToWatch1, mesh, filename1);

  // Initialize
  watchpoint0.initialize();
  watchpoint1.initialize();

  // Write output
  watchpoint0.exportPointData(0.0);
  watchpoint1.exportPointData(0.0);

  // Change geometry and write output again
  for (mesh::Vertex &vertex : mesh->vertices()) {
    BOOST_TEST(vectorValues.size() > vertex.getID());
    doubleValues[vertex.getID()] = 1.0;
  }
  watchpoint0.exportPointData(1.0);
  watchpoint1.exportPointData(1.0);
}

BOOST_AUTO_TEST_SUITE_END() // Precice
