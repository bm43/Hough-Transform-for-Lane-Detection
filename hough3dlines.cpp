// // License: see LICENSE-BSD2 (for all header files as well)
// implementation of: Iterative Hough Transform for Line Detection in 3D Point Clouds
// citation: Christoph Dalitz, Tilman Schramke, Manuel Jeltsch


#include "vector3d.h"
#include "pointcloud.h"
#include "hough.h"

#include <cstdlib>
#include <stdio.h>
#include <string.h>

#include <math.h>
#include <C:/Users/SamSung/Desktop/pers_research/Seoul_Robotics_lane_detection/Seoul_Robotics_lane_detection/hough3d-code/Eigen/Dense>

using Eigen::MatrixXf;
using namespace std;

// usage message

const char* usage = "Usage:\n"
  "\though3dlines [options] <infile>\n"
  "\t-o <outfile>   write results to <outfile> [stdout]\n"
  "\t-dx <dx>       step width in x'y'-plane [0]\n"
  "\t-nlines <nl>   maximum number of lines returned [0]\n"
  "\t-minvotes <nv> only lines with at least <nv> points are returned [0]\n";

//
// orthogonal least squares fit
// rc = largest eigenvalue
//
double orthogonal_LSQ(const PointCloud &pc, Vector3d* a, Vector3d* b){
  double rc = 0.0;

  // anchor point is mean value
  *a = pc.meanValue();

  // copy points to libeigen matrix
  Eigen::MatrixXf points = Eigen::MatrixXf::Constant(pc.points.size(), 3, 0);
  for (int i = 0; i < points.rows(); i++) {
    points(i,0) = pc.points.at(i).x;
    points(i,1) = pc.points.at(i).y;
    points(i,2) = pc.points.at(i).z;
  }


  // compute scatter matrix -> estimates the covariance matrix
  MatrixXf centered = points.rowwise() - points.colwise().mean();
  MatrixXf scatter = (centered.adjoint() * centered);

  // ... and its eigenvalues and eigenvectors
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(scatter);
  Eigen::MatrixXf eigvecs = eig.eigenvectors();

  // we need eigenvector to largest eigenvalue
  // libeigen yields it as LAST column
  b->x = eigvecs(0,2); b->y = eigvecs(1,2); b->z = eigvecs(2,2);
  rc = eig.eigenvalues()(2);

  return (rc);
}


//--------------------------------------------------------------------
// main program
//--------------------------------------------------------------------

int main(int argc, char ** argv) {

  // default values for command line options
  double opt_dx = 0.0;
  int opt_nlines = 0;
  int opt_minvotes = 0;
  enum Outformat { format_normal, format_gnuplot, format_raw };
  Outformat opt_outformat = format_normal;
  char* infile_name = NULL;
  char* outfile_name = NULL;

  // number of icosahedron subdivisions for direction discretization
  int granularity = 4;
  int num_directions[7] = {12, 21, 81, 321, 1281, 5121, 20481};

  FILE* outfile = stdout;

  // bounding box of point cloud
  Vector3d minP, maxP, minPshifted, maxPshifted;
  // diagonal of point cloud
  double d;

  // parse command line
  for (int i=1; i<argc; i++) {
    if (0 == strcmp(argv[i], "-o")) {
      i++;
      if (i<argc)
        outfile_name = argv[i];
    }
    else if (0 == strcmp(argv[i], "-dx")) {
      i++;
      if (i<argc) opt_dx = atof(argv[i]); // str->float
    }
    else if (0 == strcmp(argv[i], "-nlines")) {
      i++;
      if (i<argc) opt_nlines = atoi(argv[i]); // str->int
    }
    else if (0 == strcmp(argv[i], "-minvotes")) {
      i++;
      if (i<argc) opt_minvotes = atoi(argv[i]);
    }
    else {
      infile_name = argv[i];
    }
  }

  // plausibilty checks
  if (!infile_name) {
    fprintf(stderr, "Error: no infile given!\n%s", usage);
    return 1;
  }
  if (opt_dx < 0){
    fprintf(stderr, "Error: dx < 0!\n%s", usage);
    return 1;
  }
  if (opt_nlines < 0){
    fprintf(stderr, "Error: nlines < 0!\n%s", usage);
    return 1;
  }
  if (opt_minvotes < 0){
    fprintf(stderr, "Error: minvotes < 0!\n%s", usage);
    return 1;
  }
  if (opt_minvotes < 2){
    opt_minvotes = 2;
  }

  // open in/out files
  if (outfile_name) {
    outfile = fopen(outfile_name, "w");
    if (!outfile) {
      fprintf(stderr, "Error: cannot open outfile '%s'!\n", outfile_name);
      return 1;
    }
  }

  // read point cloud from file
  PointCloud X;
  if (0 != X.readFromFile(infile_name)) {
    fprintf(stderr, "Error: cannot open infile '%s'!\n", infile_name);
    return 1;
  }
  if (X.points.size() < 2) {
    fprintf(stderr, "Error: point cloud has less than two points\n");
    return 1;
  }

  // center cloud and compute new bounding box
  X.getMinMax3D(&minP, &maxP);
  d = (maxP-minP).norm();
  if (d == 0.0) {
    fprintf(stderr, "Error: all points in point cloud identical\n");
    return 1;
  }
  X.shiftToOrigin();
  X.getMinMax3D(&minPshifted, &maxPshifted); // reference to the value of the variable used in the main function
  // int main 에서 getMinMax3D가 쓰일때 그 안에서 이 함수의 input으로 선언된 변수들의 값을 조작할 수 있기 위해서

  // parameter space length
  if (opt_dx == 0.0) {
    opt_dx = d / 64.0;
  }
  else if (opt_dx >= d) {
    fprintf(stderr, "Error: dx too large\n");
    return 1;
  }
  double num_x = floor(d / opt_dx + 0.5);
  double num_cells = num_x * num_x * num_directions[granularity];

  // Hough Transform
  Hough* hough;
  /*
  try {
    hough = new Hough(minPshifted, maxPshifted, opt_dx, granularity);
  } catch (const std::exception &e) {
    fprintf(stderr, "Error: cannot allocate memory for %.0f Hough cells"
            " (%.2f MB)\n", num_cells,
            (double(num_cells) / 1000000.0) * sizeof(unsigned int));
    return 2;
  }
  */
  hough->add(X);

  // iterative Hough transform
  PointCloud Y;	// points close to line
  double rc;
  unsigned int nvotes;
  int nlines = 0;
  do {
    Vector3d a; // anchor point of line
    Vector3d b; // direction of line

    hough->subtract(Y);

    nvotes = hough->getLine(&a, &b);

    X.pointsCloseToLine(a, b, opt_dx, &Y);

    rc = orthogonal_LSQ(Y, &a, &b);
    if (rc==0.0) break;

    X.pointsCloseToLine(a, b, opt_dx, &Y);
    nvotes = Y.points.size();
    if (nvotes < (unsigned int)opt_minvotes) break;

    rc = orthogonal_LSQ(Y, &a, &b);
    if (rc==0.0) break;

    a = a + X.shift;

    nlines++;
    if (opt_outformat == format_normal) {
      fprintf(outfile, "npoints=%lu, a=(%f,%f,%f), b=(%f,%f,%f)\n",
              Y.points.size(), a.x, a.y, a.z, b.x, b.y, b.z);
    }
    else {
      fprintf(outfile, "%f %f %f %f %f %f %lu\n",
              a.x, a.y, a.z, b.x, b.y, b.z, Y.points.size());
    }

    X.removePoints(Y);

  } while ((X.points.size() > 1) &&
           ((opt_nlines == 0) || (opt_nlines > nlines)));

  // clean up
  delete hough;
  if (outfile_name) fclose(outfile);

  return 0;
}
