#ifndef __JOIN_H_INCLUDED__
#define __JOIN_H_INCLUDED__

#include <geos/geom/Geometry.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/Coordinate.h>
#include <geos/geom/CoordinateArraySequence.h>
#include <geos/geom/CoordinateSequenceFactory.h>
#include <geos/geom/LineString.h>
#include <geos/geom/LinearRing.h>

#include <geos/index/strtree/STRtree.h>

#include <geos/geom/prep/PreparedGeometryFactory.h>
#include <geos/geom/prep/PreparedGeometry.h>

#include<list>
#include<map>
#include <tuple>

using namespace geos::geom;
using geos::geom::Coordinate;

using namespace std;

class Join
{
  Geometry* convertEnvToGeom(Envelope *env);
  // original
  // geos::geom::GeometryFactory factory;   
 
 // buddhi added
  // geos::geom::GeometryFactory factory = geos::geom::GeometryFactory::createGeometryCollection();
  
  public:

  void joinGeomEnv(map<Geometry*, vector<void *> >*GeomToEnvsMap);
  
  list<pair<Geometry*, Geometry*> >* join(map<Geometry*, vector<void *> >* intersectionMap);
  
  long join(map<Envelope*, vector<void *> >* intersectionMap);

};

#endif 