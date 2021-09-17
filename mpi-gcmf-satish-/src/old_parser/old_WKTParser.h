#ifndef __WKTPARSER_H_INCLUDED__
#define __WKTPARSER_H_INCLUDED__

#include "Parser.h"
#include "../filePartition/FileSplits.h"
#include <geos/io/WKTReader.h>
#include <geos/geom/Geometry.h>
#include <string>
#include <tuple>

class WKTParser : public Parser 
{
  Geometry* parseString( pair<int, string> &p);
  pair<int, string> extract(const string &string);
  
  public:
  list<Geometry*>* parse(FileSplits &split);
  
  ~WKTParser() {}
   
};

#endif 