#include "parser/WKTParser.h"

pair<int, string> WKTParser :: extract(const string &str)
{
    int start = 0;

    while(start<str.length()) {
	  if(isalpha(str[start]))
		  break;
	  else
	     start++;
    }
    
    string wayIdStr = str.substr(0, start);
    int wayId = atoi(wayIdStr.c_str());
    
    int end = str.length()-1;
    while(end>0) {
	   if(str[end] == ')')
		   break;
	   else
		   end--;
    }

  //  cout<<"start "<<(start-1)<<endl;
  //  cout<<"end "<<(end-1)<<endl;
    int actual_length =  end - start + 1;

    string result = str.substr(start, actual_length);

    pair<int, string> p(wayId, result);

    return p;
}

list<Geometry*>* WKTParser :: parse(FileSplits &split) 
{
  list<Geometry*> *geoms = new list<Geometry*>();
  
  list<string>::const_iterator i;

  list<string> *contents = split.getContents();
  for(i = contents->begin(); i!= contents->end(); i++) {
    pair<int, string> p = extract(*i);
    Geometry *geom = parseString(p);
    geoms->push_back(geom);
  }
  
  return geoms;
}

Geometry* WKTParser :: parseString( std::pair<int, string> &p) 
{
  geos::io::WKTReader wktreader;
  Geometry *geom;
  
  try
  {
    geom = wktreader.read(p.second);
  
    geom->setUserData(&p.first);
  }
  catch(exception &e)
  {
    cout<< e.what() <<'\n';
  }
  
  return geom;
}