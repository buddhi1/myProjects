#include "WKTParser.h"
#include <iostream>
#include <fstream>

using namespace std;

/*
* GEOS requires two message handlers to return
* error and notice message to the calling program.
*
*   typedef void(* GEOSMessageHandler) (const char *fmt,...)
*
* Here we stub out an example that just prints the
* messages to stdout.
*/

using namespace std;

static void
geos_message_handler(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf (fmt, ap);
    va_end(ap);
}

bool WKTParser :: checkForEmptyCollection(GEOSGeometry *iGeom)
{
    bool isEmpty = false;

    // if(iGeom == NULL || iGeom->isEmpty())
    if(iGeom == NULL)
      return false;
      
    // for(int i=0; i<iGeom->getNumGeometries(); i++) {
    //    const Geometry *g = iGeom->getGeometryN(i);
      
    //    if(g->isEmpty())
    //       isEmpty = true;
    // }
    
    // return isEmpty;
}

list<GEOSGeometry*>* WKTParser:: readOSM(char* filename) {

    ifstream in_stream;
    string line;

    /* Send notice and error messages to our stdout handler */
    initGEOS(geos_message_handler, geos_message_handler);
    
    list<GEOSGeometry*> *layer;
    
    in_stream.open(filename);
  
    if (in_stream.is_open()) {
   
      layer = new list<GEOSGeometry*>();

      /* Read the WKT into geometry objects */
      GEOSWKTReader* reader = GEOSWKTReader_create();
      GEOSGeometry* geom = NULL;
      // geos::io::WKTReader wktreader;
      // Geometry *geom = NULL;

      /* Convert result to WKT */
      GEOSWKTWriter* writer = GEOSWKTWriter_create();
      /* Trim trailing zeros off output */
      GEOSWKTWriter_setTrim(writer, 1);
      // int i=0;

      for( string line; getline( in_stream, line ); )
      {
        // cout << "\nline: " << line << " " << line.find_first_of("PGLM") << endl;
        geom = GEOSWKTReader_read(reader, &line.substr(line.find_first_of("PGLM"), line.find(")")-5)[0]);
        // geom = wktreader.read(line);

        // cout << "\nline: " << line.substr(line.find("P"), line.find(")")-5) << endl;
  
        if(!checkForEmptyCollection(geom))
          layer->push_back(geom);
        // char* item1 = GEOSWKTWriter_write(writer, layer->front());

        // cout << i << item1 << "\n" << endl;
        // ++i;
        // if(i==100)
        //   break;
      }
	  in_stream.close();
     /* Clean up everything we allocated */
    GEOSWKTReader_destroy(reader);
    // GEOSGeom_destroy(geom);
   }	
	return layer;
}
