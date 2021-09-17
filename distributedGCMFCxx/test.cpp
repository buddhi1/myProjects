/* To print to stdout */
#include <stdio.h>
#include <stdarg.h>

/* Only the CAPI header is required */
#include <geos.h>

#include "WKTParser.cpp"

#include "CudaJoinInterface.h"

int main(int argc, char const *argv[])
{
	int numPolygons=0;
	CudaJoinInterface cudaInterfaceObj;
	
	WKTParser readFile;
    list<Geometry*> *layer;
    char* filename = "../data/cemetery";
    // char* filename = "test.wkt";
    layer = readFile.readOSM(filename);

    /* Convert result to WKT */
    GEOSWKTWriter* writer = GEOSWKTWriter_create();
    /* Trim trailing zeros off output */
    // GEOSWKTWriter_setTrim(writer, 1);
    // cout << layer->size() << " before" << endl;
    // char* item1 = GEOSWKTWriter_write(writer, layer->front());

    // cout << " fin: " <<  item1 << endl;

    polygonLayer* layer1Data = cudaInterfaceObj.populateLayerData(layer, &numPolygons);
	
    /* Clean up the global context */
    finishGEOS();

	return 0;
}