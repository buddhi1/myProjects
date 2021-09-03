#include <list>
#include <geos_c.h>

using namespace std;

class WKTParser 
{
    public:
        list<GEOSGeometry*>* readOSM(char* filename);
        bool checkForEmptyCollection(GEOSGeometry *iGeom);
};