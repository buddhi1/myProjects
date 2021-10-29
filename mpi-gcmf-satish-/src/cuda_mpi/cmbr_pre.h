#include <iostream>
#include <vector>
#include <array>
#include <bitset>
#include <cmath> 
#include<bits/stdc++.h>

// #include "Types.h" 

#define MAX_COUNT 5000
// #define MAX_COUNT 1000

using namespace std;

struct table_row {
    int id;
    float x;
    float y;
};

// data structure to save MBR info
struct mbr {
    float x1;
    float y1;
    float x2;
    float y2;
    bool empty = true;
};

long const DIST = 200;

int const FMAX = 13;

// number of rows in the data file
int ROWS;

int count1 = 0, count2 = 0;

// saves all counts for features
vector<int> fcount(FMAX);

vector<vector<mbr>> mbr_array(FMAX);

// used for 2 feature compariosn
// mbr_t* x_MBR1 = (mbr_t*) malloc(sizeof(mbr_t) * MAX_COUNT * 2); 
// mbr_t* x_MBR2 = (mbr_t*) malloc(sizeof(mbr_t) * MAX_COUNT * 2); 
// mbr_t* y_MBR1 = (mbr_t*) malloc(sizeof(mbr_t) * MAX_COUNT * 2); 
// mbr_t* y_MBR2 = (mbr_t*) malloc(sizeof(mbr_t) * MAX_COUNT * 2); 

mbr_t* x_MBR1; 
mbr_t* x_MBR2; 
mbr_t* y_MBR1; 
mbr_t* y_MBR2; 

// contains instances all the features
// mbr_t* x_MBR_all; 
// mbr_t* y_MBR_all; 

// prefix sum of fcount
int prefixSumFcount[FMAX];

// coord_t* seq_oMBR2 = (coord_t*) malloc(sizeof(coord_t) * MAX_COUNT * 4); 
// coord_t* seq_bMBR2 = (coord_t*) malloc(sizeof(coord_t) * MAX_COUNT * 4); 


void print_message(string str) {
    if (true)
    {
        cout << str << endl;        
    }
}

// read file data and retunr data as an object array
struct table_row *createArray(const char *fileName) {
    FILE *fp = fopen(fileName, "r");

    if (!fp) {
        print_message("Can not open file\n");
        return NULL;
    }
    
    fscanf(fp, "%d", &ROWS);
    // cout << "Total rows: " << ROWS << endl;

    struct table_row* table_rows = (struct table_row*)malloc(sizeof(struct table_row) * ROWS);
    
    for (int count = 0; count < ROWS; ++count) {
        fscanf(fp, "%d, %f, %f", &table_rows[count].id, &table_rows[count].x, &table_rows[count].y);

    }
    
    fclose(fp);
    return table_rows;
}

// calculate MBR for a given datapoint
mbr getMBR(float px, float py) {
    mbr box;
    box.x1 = px - DIST;
    box.y1 = py - DIST;
    box.x2 = px + DIST;
    box.y2 = py + DIST;

    return box;
}

void getMBRList(struct table_row *data) {

    int j = 0, k = 0;
    int row, col;
    mbr temp;

    print_message("Grid set Start...");
    for (int i = 0; i < ROWS; ++i) { 
        // check if feature id changes
        if (j != (data[i].id - 1)) {
            j = (data[i].id - 1);
            k++;
        }

        temp = getMBR(data[i].x, data[i].y);

        // calculate MBR using the getMBR() and assign it to the relavant feature instance
        mbr_array[k].push_back(temp);
        // cout << k << mbr_array[k].x1;
        fcount[k] += 1; 
    }
    print_message("Grid set...");
}

// prefix sum of the features sizes
void prefixSumSizes() {
    prefixSumFcount[0] = fcount[0];
    for (int i = 1; i < FMAX; ++i)
    {
        prefixSumFcount[i] = prefixSumFcount[i-1] + fcount[i];
    }
}

// defines size for MBR arrays
// void setMBRArrays() {
//     for (int featureID = 0; featureID < FMAX; ++featureID)
//     {
//         x_MBR_all = (mbr_t*) malloc(sizeof(mbr_t) * fcount[featureID] * 2 * FMAX);         
//         y_MBR_all = (mbr_t*) malloc(sizeof(mbr_t) * fcount[featureID] * 2 * FMAX);         
//     }
// }

void createMBRArrays(int layer2ID) {
    x_MBR1 = (mbr_t*) malloc(sizeof(mbr_t) * (prefixSumFcount[FMAX-1]-fcount[layer2ID]) * 2); 
    y_MBR1 = (mbr_t*) malloc(sizeof(mbr_t) * (prefixSumFcount[FMAX-1]-fcount[layer2ID]) * 2); 
    x_MBR2 = (mbr_t*) malloc(sizeof(mbr_t) * fcount[layer2ID] * 2); 
    y_MBR2 = (mbr_t*) malloc(sizeof(mbr_t) * fcount[layer2ID] * 2); 
}

// get number of digits of a given number
int getDigitCount(long num) {
    int count = 0;
    while(num != 0) {
        num /= 10;
        count++;
    }
    return count;
}

// get how many decimal points
long convertFloatToLong(float num, int id, int i) {
    long x;
    int n;

    if (mbr_array[id].size() <= i)
    {
        x = 0; 
    } else {    
        x = num*100;
    }
    // cout << id << " " << mbr_array[id].size() << " " << i << " " << x << endl;

    return x;
}

// populate seq arrays
// void populateSeqArrays(int fid1, int fid2) {
//     int c = 0;
//     for (int i = 0; i < count1; i++)
//     {
//         seq_bMBR2[c*4] = mbr_array[fid1][i].x1;
//         seq_bMBR2[c*4+1] = mbr_array[fid1][i].y1;
//         seq_bMBR2[c*4+2] = mbr_array[fid1][i].x2;
//         seq_bMBR2[c*4+3] = mbr_array[fid1][i].y2;
//         c++;
//     }

//     c = 0;
//     for (int i = 0; i < count2; i++)
//     {
//         seq_oMBR2[c*4] = mbr_array[fid2][i].x1;
//         seq_oMBR2[c*4+1] = mbr_array[fid2][i].y1;
//         seq_oMBR2[c*4+2] = mbr_array[fid2][i].x2;
//         seq_oMBR2[c*4+3] = mbr_array[fid2][i].y2;
//         c++;
//     }
// }

// used for 2 feature comparision only
// select smaple data
// void preProcessMBRArray(int fid1, int fid2) {
//     count1 = 0, count2 = 0;

//     // for (int i = 0; i < /*3190*/mbr_array[fid1].size(); ++i)
//     for (int i = 0; i < MAX_COUNT /* 3190*/; ++i)
//     {
//         x_MBR1[count1*2] = convertFloatToLong(mbr_array[fid1][i].x1, fid1, i);
//         x_MBR1[count1*2+1] = convertFloatToLong(mbr_array[fid1][i].x2, fid1, i);
//         y_MBR1[count1*2] = convertFloatToLong(mbr_array[fid1][i].y1, fid1, i);
//         y_MBR1[count1*2+1] = convertFloatToLong(mbr_array[fid1][i].y2, fid1, i);

//         count1++;
//     }
//     // for (int i = 0; i < /*3899*/mbr_array[fid2].size(); ++i)
//     for (int i = 0; i < MAX_COUNT /* 6000*/; ++i)
//     {

//         x_MBR2[count2*2] = convertFloatToLong(mbr_array[fid2][i].x1, fid2, i);
//         x_MBR2[count2*2+1] = convertFloatToLong(mbr_array[fid2][i].x2, fid2, i);
//         y_MBR2[count2*2] = convertFloatToLong(mbr_array[fid2][i].y1, fid2, i);
//         y_MBR2[count2*2+1] = convertFloatToLong(mbr_array[fid2][i].y2, fid2, i);

//         count2++;
//     }
//     // populateSeqArrays(fid1, fid2);
// }

// converts all MBR data to x_MBR_all and y_MBR_all arrays
// void preProcessAllMBRArray() {

//     // set arrays without having extra space
//     setMBRArrays();

//     // x_MBR_all = (mbr_t*) malloc(sizeof(mbr_t) * MAX_COUNT * 2 * FMAX);         
//     // y_MBR_all = (mbr_t*) malloc(sizeof(mbr_t) * MAX_COUNT * 2 * FMAX); 

//     for (int j = 0; j < FMAX; ++j)
//     {
//         for (int i = 0; i < MAX_COUNT; ++i)
//         {
//             x_MBR_all[prefixSumFcount[j]-fcount[j] + i*2] = convertFloatToLong(mbr_array[j][i].x1, j-1, i);
//             x_MBR_all[prefixSumFcount[j]-fcount[j] + i*2+1] = convertFloatToLong(mbr_array[j][i].x2, j-1, i);
//             y_MBR_all[prefixSumFcount[j]-fcount[j] + i*2] = convertFloatToLong(mbr_array[j][i].y1, j-1, i);
//             y_MBR_all[prefixSumFcount[j]-fcount[j] + i*2+1] = convertFloatToLong(mbr_array[j][i].y2, j-1, i);
//         }
//     }
// }


// converts all MBR data to mbr arrays. 
// Given feature will be excluded from layer 1 and included in layer 2
void preProcessTO2Layers(int layer2ID) {
    // set arrays without having extra space
    createMBRArrays(layer2ID);
    count1 = 0, count2 = 0;


    // prepare layer 1 
    // for (int j = 0; j < FMAX; ++j)
    for (int j = 0; j < 4; ++j)
    {
        if (j == layer2ID)
        {
            continue;
        }
        for (int i = 0; i < fcount[j] /* 3190*/; ++i)
        {
            x_MBR1[count1*2] = convertFloatToLong(mbr_array[j][i].x1, j, i);
            x_MBR1[count1*2+1] = convertFloatToLong(mbr_array[j][i].x2, j, i);
            y_MBR1[count1*2] = convertFloatToLong(mbr_array[j][i].y1, j, i);
            y_MBR1[count1*2+1] = convertFloatToLong(mbr_array[j][i].y2, j, i);

            count1++;
        }
    }

    // prepare layer 2
    for (int i = 0; i < fcount[layer2ID] /* 6000*/; ++i)
    {
        x_MBR2[count2*2] = convertFloatToLong(mbr_array[layer2ID][i].x1, layer2ID, i);
        x_MBR2[count2*2+1] = convertFloatToLong(mbr_array[layer2ID][i].x2, layer2ID, i);
        y_MBR2[count2*2] = convertFloatToLong(mbr_array[layer2ID][i].y1, layer2ID, i);
        y_MBR2[count2*2+1] = convertFloatToLong(mbr_array[layer2ID][i].y2, layer2ID, i);

        count2++;
    }
}

void printFeatureArray(mbr_t *x, int start, int count) {
    for (int i = 0; i < count; ++i)
    {
        cout << x[prefixSumFcount[start]-fcount[start] + i] << " ";
    }
    cout << endl;
}

void printArray(mbr_t *x, int count) {
    for (int i = 0; i < count; ++i)
    {
        cout << x[i] << " ";
    }
    cout << endl;
}

void printArray_coord_t(coord_t *x, int count) {
    for (int i = 0; i < count; ++i)
    {
        cout << x[i] << " ";
    }
    cout << endl;
}

// int main() {
//     // read data into a table_row structure type 1D array
//     struct table_row *dat;
//     dat = createArray("data/Point_Of_Interest_modified.csv");

//     getMBRList(dat);
//     print_message("mbr array constructed");

//     prefixSumSizes();
//     preProcessTO2Layers(1);

//     cout << "preprocess done" << endl;
//     // cout<<"Numbers after decimal point = "<<getNumDecimalDigits(12.351)<<endl; 
//     printArray(x_MBR1, 10);
//     printArray(y_MBR1, 10);
//     printArray(x_MBR2, 10);
//     printArray(y_MBR2, 10);
   
//    // convertFloatToLong(mbr_array[0][0].x1);
   
//     return 0;
// }