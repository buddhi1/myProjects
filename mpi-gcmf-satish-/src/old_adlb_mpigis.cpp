#include <string>
#include <list>
#include "filePartition/FilePartitioner.h"
#include "filePartition/MPI_File_Partitioner.h"
#include "filePartition/MPI_LargeFile_Partitioner.h"

#include "filePartition/FileSplits.h"
#include "parser/Parser.h"
#include "parser/WKTParser.h"
#include "filePartition/config.h"
#include <geos/geom/Geometry.h>
#include "spatialPartition/grid.h"
#include "spatialPartition/uniformGrid.h"
#include "index/index.h"
#include "join/join.h"
#include "overlay/overlay.h"
#include <fstream>
#include <iostream>
#include <cstdlib>

#include "mpiTypes/mpitype.h"
#include "geom_util/util.h"
#include "parser/road_network_parser.h"
#include "spatialPartition/uniformGrid.h"
#include "spatialPartition/RtreeStructure.h"
#include "taskMap/TaskMap.h"
#include "taskMap/roundRobinTaskMap.h"

#include "bufferManager/singleLayerGeomBufferMgr.h"
#include "geom_util/fileRead.h"

#include "bufferManager/bufferManagerGeoms.h"
#include "mapreduce/mrdriver.h"

#include "bufferManager/bufferRoadnetwork.h"
#include "bufferManager/MultiRoundOneLayerGeomBufferMgr.h"

#include "bufferManager/AllObjectsCommBuffer_DerivedType.h"
#include "parser/pointNodeParser.h"

#include <fstream>
#include <unistd.h>

//#define DBUG2 2
#define DBUG1 1
#define BREAKDOWN 3

using namespace std;

// /home/satish/mpigis/indexed_data/layer1
// /home/satish/mpigis/indexed_data/layer2

//mpirun -np 2 ./adlb 8 /home/satish/mpigis/indexed_data/layer1 /home/satish/mpigis/indexed_data/layer2

int main(int argc, char **argv)
{	
    Config conf(argc, argv); 
    
    conf.initMPI(argc, argv);

    double total_t1;
	
	char hostname[256];
	gethostname(hostname,255);
	
	total_t1 = MPI_Wtime();
	
	vector<string>* layer1 = conf.getLayer1();
	
    vector<string>* layer2 = conf.getLayer2();

    int rank = conf.rank;
    int numProcesses = conf.numProcesses;
    int numFiles = conf.numPartitions;
    cout<<"#indexed files "<<numFiles<<", #processes "<<numProcesses<<endl;
    
    // 2 is hard-coded because the first two file are hidden file "." and ".."
    for(int fileNum = rank+2; fileNum < (numFiles+2); fileNum = fileNum + numProcesses) {
    
    	string strFileLayer1 = layer1->at(fileNum);
    	char *fileLayer1 = const_cast<char*>(strFileLayer1.c_str());
    
    	string strFileLayer2 = layer2->at(fileNum);
    	char *fileLayer2 = const_cast<char*>(strFileLayer2.c_str());
    
		cout<<fileLayer1<<endl;
    	cout<<fileLayer2<<endl;
    
    	int numGridFiles = conf.numPartitions;
    
    	
    	// first two files in a directory are "." and ".."
    
    	Parser *parser = new WKTParser();
    
    	FileReader fileReader;
    	list<Geometry*> *geomsLayer1 = fileReader.readSampleFile(fileLayer1, &conf);
    	cerr<<"P"<<conf.rank<<" "<<hostname<<", Layer1 File #"<<(fileNum-2)<<", geoms, "<<geomsLayer1->size()<<endl;
    	
    	Index index;
        index.createRTree(geomsLayer1);
    
   		list<Geometry*> *geomsLayer2 = fileReader.readSampleFile(fileLayer2, &conf);
	    cerr<<"P"<<conf.rank<<" "<<hostname<<", Layer2 File #"<<(fileNum-2)<<", geoms, "<<geomsLayer2->size()<<endl;	
	    
	    map<Geometry*, vector<void *> >*joinResult = index.query(geomsLayer2);
    
        Join joinObj;
    
        list<pair<Geometry*, Geometry*> >*pairs = joinObj.join(joinResult);
        cout<<"Output Pairs "<<pairs->size()<<endl;
     }
     
    double total_t2 = MPI_Wtime();
  
    if(conf.rank == 0)
	   cout<<"Total time "<<(total_t2 - total_t1)<<endl;
  
    MPI_Finalize();
}


int main1(int argc, char **argv)
{	

	Config args(argc, argv);
	
	args.initMPI(argc, argv);
	
	double t1, t2;
	
	t1 = MPI_Wtime();
	
	FilePartitioner *partitioner = new MPI_File_Partitioner();

	partitioner->initialize(args);
	//cout<<"init done"<<endl;
    
    pair<FileSplits*, FileSplits*> splitPair = partitioner->partition();

	Parser *parser = new WKTParser();
	
	cout<<"P"<<args.rank<<" lines, "<<splitPair.first->numLines()<<endl;
	
	list<Geometry*> *layer1Geoms = parser->parse(*splitPair.first);

    //Overlay overlay;
    //overlay.initOutputFile(args->rank);

    //overlay.writeGeoms(layer1Geoms);

    //overlay.closeFile();
    
    //if(layer1Geoms == NULL)
    //    cout<<args->rank<<": NULL geometries "<<endl;
    //else
    	cout<<"P"<<args.rank<<", geoms, "<<layer1Geoms->size()<<endl;

	splitPair.first->clear();
    
    delete splitPair.first;
	
	
	Index tree;
	tree.createRTree(layer1Geoms);
	
	//tree.createIndex(layer1Geoms);
	cout<<args.rank<<", Index created "<<endl;
	
	FileSplits *layer2 = new FileSplits();
    
    ifstream in_stream;
    string line;
    
    in_stream.open(argv[3]);

    for( string line; getline( in_stream, line ); )
    {
      layer2->addGeom(line);
    }
	
	//cout<<"P"<<args->rank<<" Number of lines in layer2 "<<layer2->numLines()<<endl;
	
	Parser *parser2 = new WKTParser();
    
    list<Geometry*> *layer2Geoms = parser2->parse(*layer2);
	cout<<"P"<<args.rank<<" Layer 2 geoms, "<<layer2Geoms->size()<<endl;
	
	
	map<Geometry*, vector<void *> >* intersectionMap = tree.query(layer2Geoms);
	
	//if(intersectionMap!=nullptr)
    //	cerr<<args->rank<<" : query result "<<intersectionMap->size()<<endl;
	
	
	map<Geometry*, vector<void *> >::iterator itOuter;
    int numOverlaps = 0;   
    for (itOuter = intersectionMap->begin(); itOuter != intersectionMap->end(); ++itOuter) {
           // std::cout << it->first << " => " << it->second->size() << '\n';
         numOverlaps += itOuter->second.size();  
    }
    cerr<<args.rank<<", query result, "<<numOverlaps<<endl;
    
    Join spatialJoin;

    list<pair<Geometry*, Geometry*> > *pairs = spatialJoin.join(intersectionMap);
    
    /*Overlay overlay;
    list<pair<Geometry*, Geometry*> > *pairs = overlay.overlay(intersectionMap);	
    */
	
    //cout<<args->rank<<", Number of overlapping pairs "<<pairs->size()<<endl;
		
	/*Grid *grid = new UniformGrid(16);
	
    grid->setDimension(1,2,3,4);
    
	grid->populateGridCells(*layer1Geoms, true);
	
	grid->populateGridCells(*layer2Geoms, false);

	//shuffleExchange();
	*/
	partitioner->finalize();
	
	//delete parser;
	//delete partitioner;

    t2 = MPI_Wtime();
   
    cout<<"Time ,"<<args.rank<<", "<<(t2-t1)<<endl;

	MPI_Finalize();

	return 0;
}
