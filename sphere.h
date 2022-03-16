// citation: Hough Parameter Space Regularisation for Line Detection in 3D, Jeltsch, Dalitz, Pohle-Froehlich

#ifndef SPHERE_H_
#define SPHERE_H_

#include "vector3d.h"
#include <vector>
#include <deque>

class Sphere {

public:
  // direction vectors
  std::vector<Vector3d> vertices;
  // surface triangles
  std::deque<unsigned int> triangles;
  // creates the directions by subdivisions of icosahedron
  void fromIcosahedron(int subDivisions=4);

private:
  // creates nodes and edges of icosahedron
  void getIcosahedron();
  // one subdivision step
  void subDivide();
  // make vectors nondirectional and unique
  void makeUnique();

};

#endif /* SPHERE_H_ */
