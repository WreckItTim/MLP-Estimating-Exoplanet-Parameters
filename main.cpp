/*
 README
 
 The purpose of this code is to provide a lightweight implementation of training new MLP's
 
 A paper describing the methods can be found here:
 
 Open any .csv data file in our GitHub for an example on how to format data fed into the MLP.
 Data files to be used by the MLP need to follow this format:
 one .csv file
 first row -> header ("Sample", Wavelengths... (wavelengths must be purely numerical values, example "54.1"), Label Names... (label names must be string values not purely numerical but they can contain numbers, example "Label1" is ok by "1" is not)
 subsequent rows -> Sample Name, Intensity Values..., Label Values...
 
 To use this code, simply run and specify paths of a data .csv file for training, a data .csv file for validation, and a data .csv file for testing. If you enter 'auto' when prompted, the program will autodetect the number of input nodes and output nodes to make a new MLP. Results will be output to the same folder that the training data file is saved to on your computer. Results consist of a subfolder with the MLP and predictions on the validation set after each epoch, the MLP written as a .csv file at the epoch with the lowest validation RMSE next to your data file, and the predictions on your test set using that MLP with lowest validation RMSE.
 
 The default MLP used when you enter 'auto' follows the Greedy0 MLP as presented in the paper. This is 3 hidden layers with 64-64-32 nodes in the layers, using ELU activation function; an output layer using sigmoid activation function, the labels in your data files will automatically be normalized; L1 of 1E-7; L2 of 1E-5; Max-Norm radius of 6.1; a static learning momentum of 0.9; a cyclic learning rate that cycles between 0.1 and 0.9 with a triangular cycle and step-size of 2*Epoch; trains up to 2048 epochs; is a regressor; outputs MLP, validation RMSE, and test predictions for each epoch.
 
 If you want to change the MLP parameters, optionally type 'own' instead of 'auto' into the terminal when it asks which you want to use. This will trigger a secret sub-menu which will walk you through each hyper-parameter of the MLP.
 */

#include <iostream>
#include <string>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <random>
#include <fstream>
#include <stdio.h>
#include <cstdio>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>
using namespace std;

// change this to match the number of rows and columns in your images (if using images)
bool canMultiThread = true; // multithread speeds up train time by threading sets of vectors used in each batch, and test time by threading sets of vectors used during prediction
int nGlobalThreads = 4; // OPTIMIZE THIS FOR YOUR COMPUTER, it is the optimal number of threads you can use. Number of local threads during runtime are calculated based on this global number and can be further optimized, but not necesary...

/*
 Stopwatch used to measure how much time certain events in the code take
 */
struct StopWatch {
    long last = 0;
    long start = 0;
    
    /* create a timer, and start counting */
    StopWatch() {
        auto now = chrono::system_clock::now();
        auto now_ms = chrono::time_point_cast<std::chrono::milliseconds>(now);
        auto epoch = now_ms.time_since_epoch();
        auto value = chrono::duration_cast<std::chrono::milliseconds>(epoch);
        start = value.count();
    }
    
    /* stop timer and return elapsed time from constructor */
    long stop() {
        auto now = std::chrono::system_clock::now();
        auto now_ms = chrono::time_point_cast<std::chrono::milliseconds>(now);
        auto epoch = now_ms.time_since_epoch();
        auto value = chrono::duration_cast<std::chrono::milliseconds>(epoch);
        long end = value.count();
        return end - start;
    }
    
    /* pause timer and get elapsed time from either constructor or last lap (which ever happened last) */
    long lap() {
        long time = stop();
        long delta = time - last;
        last = time;
        return delta;
    }
};

/*
    checks if character is a digit
    */
bool isDigit(char c) {
    if (c == '.' || c == '1' || c == '2' || c == '3' || c == '4' || c == '5' || c == '6' || c == '7' || c == '8' || c == '9' || c == '0')
        return true;
    return false;
}

/*
    reads a string from user
 */
string readString() {
    string str = "";
    cin >> str;
    cin.clear();
    cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return str;
}

/*
 returns a vector<string> of tokens separated from a passed in string for a given char delimiter
 */
vector<string> split(const string& s, const char& delimiter) {
    vector<string> ret; string next = "";
    
    // check if string is empty
    if (s.size() == 0) return ret;
    
    // if not empty, then go through string and separate into blocks
    for (int i = 0; i < s.size(); i++) {
        // check if char at index i is the delimiter
        if (s.at(i) == delimiter) {
            // add currently built string to vector<string> ret for later return
            if (next != "")
                ret.push_back(next);
            
            // clear currently built string for next block
            next = "";
        }
        // if char at index i is no delimiter, continue building string
        else if(s.at(i) != '\n' && s.at(i) != '\r' && s.at(i) != '\t') next += s.at(i);
    }
    
    // add last block to vector<string> ret for later return
    if (!next.empty()) ret.push_back(next);
    
    return ret;
}

/*
 OS dependent stuff
 */
string getFileName(const string& path, const bool& withExt);
bool isFile(const string& path);
bool isFolder(const string& path);
vector<string> getFilesInFolder(const string& path, const bool& withExt);
vector<string> getFoldersInFolder(const string& path);
void createFolder(string name);

#ifdef _WIN32
//define something for Windows (32-bit and 64-bit)
#include <filesystem>
#include <direct.h>
namespace fs = std::experimental::filesystem;

/*
 returns the file name at a passed in string path, withExt = true returns the file extension and false does not
 */
string getFileName(const string& path, const bool& withExt) {
    // split path into blocks, seperated by a backslash
    vector<string> data = split(path, '\\');
    
    // file name is last block
    string fileName = data.back();
    
    // chop off extension if asked to not have it
    if (!withExt)
        fileName = fileName.substr(0, fileName.size() - 4);
    
    // last block is file name
    return fileName;
}

/*
 eturns true if path points to a file
 */
bool isFile(const string& path) {
    struct stat s;
    return (stat(path.c_str(), &s) == 0 && s.st_mode & S_IFREG);
}

/*
 returns true if path points to a folder
 */
bool isFolder(const string& path) {
    struct stat s;
    return (stat(path.c_str(), &s) == 0 && s.st_mode & S_IFDIR);
}

/*
 gets file names of all valid files in given folder
 */
vector<string> getFilesInFolder(const string& path, const bool& withExt) {
    vector<string> ret;
    for (auto & p : fs::directory_iterator(path))
        if (isFile(p.path().string())) {
            string fileName = getFileName(p.path().string(), withExt);
            if (fileName == "desktop" || fileName == "desktop.ini")
                continue;
            ret.push_back(fileName);
        }
    return ret;
}

/*
 gets folder names of all valid folders in given folder
 */
vector<string> getFoldersInFolder(string path) {
    vector<string> ret;
    for (auto & p : fs::directory_iterator(path))
        if (isFolder(p.path().string()))
            ret.push_back(split(p.path().string(), '\\').back());
    return ret;
}

/*
 creates a folder a specified path
 */
void createFolder(string path) {
    _mkdir(path.c_str());
}
#ifdef _WIN64
//define something for Windows (64-bit only)
#else
//define something for Windows (32-bit only)
#endif
#elif __APPLE__
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR
// iOS Simulator
#elif TARGET_OS_IPHONE
// iOS device
#elif TARGET_OS_MAC
// define something for other mac OS
#include <dirent.h>
#include <sys/stat.h>
/*
 returns the file name at a passed in string path, withExt = true returns the file extension and false does not
 */
string getFileName(const string& path, const bool& withExt) {
    // split path into blocks, seperated by a backslash
    vector<string> data = split(path, '\\');
    
    // file name is last block
    string fileName = data.back();
    
    // chop off extension if asked to not have it
    if (!withExt)
        fileName = fileName.substr(0, fileName.size() - 4);
    
    // last block is file name
    return fileName;
}

/*
 returns true if path points to a file
 */
bool isFile(const string& path) {
    struct stat s;
    return (stat(path.c_str(), &s) == 0 && s.st_mode & S_IFREG);
}

/*
 returns true if path points to a folder
 */
bool isFolder(const string& path) {
    struct stat s;
    return (stat(path.c_str(), &s) == 0 && s.st_mode & S_IFDIR);
}

/*
 gets file names of all valid files in given folder
 */
vector<string> getFilesInFolder(const string& path, const bool& withExt) {
    vector<string> ret;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (path.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            string name = ent->d_name;
            if(name == "desktop.ini")
                continue;
            if(ent->d_type == DT_REG && name.size() >= 1 && name.at(0) != '.') {
                if(!withExt) {
                    for(unsigned long c = name.size() - 1; c >= 0; c--)
                        if(name.at(c) == '.') {
                            name = name.substr(0, c);
                            break;
                        }
                }
                ret.push_back(name);
            }
        }
        closedir (dir);
    }
    return ret;
}

/*
 gets folder names of all valid folders in given folder
 */
vector<string> getFoldersInFolder(const string& path) {
    vector<string> ret;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (path.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            string name = ent->d_name;
            if(name == "desktop.ini")
                continue;
            if(ent->d_type == DT_DIR && name.size() >= 1 && name.at(0) != '.')
                ret.push_back(name);
        }
        closedir (dir);
    }
    return ret;
}

/*
 creates a folder a specified path
 */
void createFolder(string path) {
    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}
#else
#   error "Unknown Apple platform"
#endif
#elif __linux__
// linux
#elif __unix__ // all unices not caught above
// Unix
#elif defined(_POSIX_VERSION)
// POSIX
#else
#   error "Unknown compiler"
#endif

/*
 class used to store a data set
 it saves raw data, labels, predictions, classifications, and any reconstructions if using an RBM
 WARNING: it is up to the programmer to delete header and labelNames as these are typically shared by multiple data sets! Repeat, the "header" and "labelNames" array will not be deleted with the DataSet destructor.
 */
struct DataSet {

    string* samples = NULL;

    double* header = NULL; // for spectra, this is usually wavenumbers
    int nCols = 0;
    int nRows = 0;
    double* data = NULL;

    string* labelNames = NULL;
    int nLabels = 0;
    double* labels = NULL;

    double* predictions = NULL;
    string* binaryClass = NULL;
    int* multiClass = NULL;

    /*
        DataSet_Constructor1() creates a subset of another dataset
        @param1 baseSet to pull data from
        @param2 list of rows to pull
        @param3 optional parameter to include noisy copies:
                .at(0) = number of copies (e.g. if you want 10 noisy models set this equal to 10)
                .at(1) = level of noise (e.g. if you want 5% noise set this to 0.05)
    */

    DataSet(DataSet* baseSet, vector<int> rows, vector<double> noise) {

        // check param valididity
        if (baseSet == NULL)
            cout << "@param1 in DataSet_Constructor1() is NULL" << endl;
        if (rows.empty())
            cout << "@param2 in DataSet_Constructor1() is empty" << endl;
        if (noise.size() != 0 && noise.size() != 2)
            cout << "@param3 in DataSet_Constructor1() is invalid size, must be 0 or 2" << endl;

        // create sample names array
        int nCopies = noise.empty() ? 1 : noise.at(0);
        nRows = rows.size() * nCopies;
        samples = new string[nRows];

        // create label names array
        nLabels = baseSet->nLabels;
        labelNames = new string[nLabels];
        for (int l = 0; l < nLabels; l++)
            labelNames[l] = baseSet->labelNames[l];

        // create header array
        nCols = baseSet->nCols;
        header = new double[nCols];
        for (int c = 0; c < nCols; c++)
            header[c] = baseSet->header[c];

        // fill data by pulling rows
        data = new double[nRows * nCols];
        labels = new double[nRows * nLabels];
        int r = 0;
        for(int i = 0; i < rows.size(); i++) {
            int row = rows.at(i);
            // copy data over
            for (int m = 0; m < nCopies; m++, r++) {
                samples[r] = baseSet->samples[row];
                if (nCopies > 1)
                    samples[r] += "_#" + to_string(m+1);
                for (int c = 0; c < nCols; c++)
                    data[r * nCols + c] = baseSet->data[row * nCols + c];
                for (int l = 0; l < nLabels; l++)
                    labels[r * nLabels + l] = baseSet->labels[row * nLabels + l];
            }
        }

        // add noise
        if (!noise.empty())
            addNoise(noise.at(1));
    }

    /*
     make sure to clear memory when this object is deleted
     */
    ~DataSet() {
        delete[] labelNames;
        delete[] header;
        if (data)
            delete[] data;
        if (samples)
            delete[] samples;
        if (labels)
            delete[] labels;
        if (predictions)
            delete[] predictions;
    }
    
    /*
     adds gaussian noise to data, deviated around the passsed i noise level (such as 0.05, 0.10, 0.20, etc...
     */
    void addNoise(double level) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine rng (seed);
        for (int r = 0; r < nRows; r++) {
            
            // get the average for this row
            double mean = 0.0;
            for (int c = 0; c < nCols; c++)
                mean += data[r * nCols + c];
            mean /= ((double)nCols);
            
            // make gaussian noise
            normal_distribution<double> normallyDistributed(0.0, mean * level);
            
            // add noise to each data point
            for (int c = 0; c < nCols; c++)
                data[r * nCols + c] = min(1.0, max(0.0, data[r * nCols + c] + normallyDistributed(rng)));
        }
    }
    
    /*
     writes header, data, sample names, labels, and any predictions to path with comma delims
     parameter is name of file (without file extension, assumes it is a .csv as that is what is used for writing)
     */
    void write(const string& path) {
        // open file for writing
        ofstream out(path + ".csv");
        
        // write header
        if(samples)
            out << "Model";
        for (int c = 0; c < nCols; c++)
            out << "," << header[c];
        for (int c = 0; c < nLabels; c++) {
            out << "," << labelNames[c] ;
            if (predictions)
                out << ",Prediction";
        }
        if(binaryClass)
            out << ",BinaryClass";
        if(multiClass)
            out << ",MultiClasses";
        out << "\n";
        
        // write data
        for (int r = 0; r < nRows; r++) {
            if(samples)
                out << samples[r];
            for (int c = 0; c < nCols; c++)
                out << setprecision(7) << ","  << data[r * nCols + c];
            for (int c = 0; c < nLabels; c++) {
                out << "," << labels[r * nLabels + c];
                if (predictions)
                    out << "," << predictions[r * nLabels + c];
            }
            if(binaryClass)
                out << "," << binaryClass[r];
            if(multiClass) {
                int mCounter = 0;
                for (int c = 0; c < nLabels; c++)
                    if(multiClass[r * nLabels + c] == 1) {
                        if(mCounter >= 1)
                            out << " & ";
                        out << labelNames[c];
                        mCounter++;
                    }
            }
            out << "\n";
        }
        
        out.close();
    }

    /*
        default constructor
    */
    DataSet() {

    }

    /*
        reads all data from folder
    */
    DataSet(string modelFolder) {
        // create matrix to be turned into a DataSet
        map<string, pair<vector<double>, vector<double>> > models; // map<modelName, pair<modelLabels, modelAlbedos>>

        // iterate through all model folders to extract files
        vector<string> mFolders = getFoldersInFolder(modelFolder);
        for (int i = 0; i < mFolders.size(); i++) {
            string mFolder = mFolders.at(i);

            vector<string> gFolders = getFoldersInFolder(modelFolder + "/" + mFolder + "/");
            for (int j = 0; j < gFolders.size(); j++) {
                string gFolder = gFolders.at(j);

                vector<string> tFolders = getFoldersInFolder(modelFolder + "/" + mFolder + "/" + gFolder + "/");
                for (int k = 0; k < tFolders.size(); k++) {
                    string tFolder = tFolders.at(k);

                    vector<string> fFiles = getFilesInFolder(modelFolder + "/" + mFolder + "/" + gFolder + "/" + tFolder + "/", false);
                    for (int l = 0; l < fFiles.size(); l++) {
                        string fFile = fFiles.at(l);

                        // get parts of this file, and subsequent labels
                        vector<string> parts = split(fFile, '_');
                        double t = stod(parts.at(4).substr(1));
                        double g = stod(parts.at(3).substr(1));
                        double m = stod(parts.at(2).substr(1));
                        double f = stod(parts.at(5).substr(1));
                        vector<double> modelLabels = { m, g, t, f };

                        // get albedo values for this file, and header if we have not yet
                        vector<double> modelWavelengths;
                        vector<double> modelAlbedos;
                        ifstream in(modelFolder + "/" + mFolder + "/" + gFolder + "/" + tFolder + "/" + fFile + ".csv");
                        string line = "";
                        while (getline(in, line)) {
                            parts = split(line, ',');
                            double wavelength = stod(parts.at(0)); // microns
                            double albedo = stod(parts.at(1)); // ratio of reflectance at this wavelength [0,1]
                            modelAlbedos.push_back(albedo);
                            modelWavelengths.push_back(wavelength);
                        }
                        
                        // fill header if we have not yet
                        if (!header) {
                            header = new double[modelWavelengths.size()];
                            nCols = modelWavelengths.size();
                        }
                        for (int i = 0; i < modelWavelengths.size(); i++)
                            header[i] = modelWavelengths.at(i);

                        // save to models map
                        models[fFile] = pair<vector<double>, vector<double>>{ modelLabels, modelAlbedos };
                    }
                }
            }
        }

        // make data and label values for this object
        nRows = models.size();
        samples = new string[nRows];
        labelNames = new string[4];
        labelNames[0] = "[M/H]";
        labelNames[1] = "log g(cm_s2)";
        labelNames[2] = "T_eff(K)";
        labelNames[3] = "f_sed";
        nLabels = 4;
        labels = new double[4 * nRows];
        data = new double[nRows * nCols];
        int r = 0;
        for (map<string, pair<vector<double>, vector<double>> >::iterator it = models.begin(); it != models.end(); it++, r++) {
            samples[r] = it->first;
            for (int l = 0; l < nLabels; l++)
                labels[r * nLabels + l] = it->second.first.at(l);
            for (int c = 0; c < nCols; c++)
                data[r * nCols + c] = it->second.second.at(c);
        }
    }
    
    
    /*
     Reads sample name, data values, and label values from one .txt or .csv file
     first paramter is full path to file
     second parameter is NULL, unless the user wants to interpolate this DataSet values to the passed in DataSet when using a .txt file
     */
    DataSet(string path, DataSet* interpolateTo) {
        cout << "reading data from: '" << path << "'" << endl;
        
        // read .txt spectrum generated from the berkeley people
        if(path.substr(path.size() - 4) == ".txt") {
            ifstream in(path);
            string line = "";
            getline(in, line);
            map<double, double> values;
            double minW = -1.0;
            double maxW = -1.0;
            while(getline(in, line)) {
                vector<string> tokens = split(line, ' ');
                double wavelength = stod(tokens.at(0));
                double intensity = stod(tokens.at(1));
                //double cratio = stod(tokens.at(2));
                //double upper = stod(tokens.at(3));
                //double downer = stod(tokens.at(4));
                
                values[wavelength] = intensity;
                
                if(minW == -1.0)
                    minW = wavelength;
                maxW = wavelength;
            }
            
            // initialize variables
            nCols = interpolateTo->nCols;
            nRows = 1;
            nLabels = interpolateTo->nLabels;
            header = new double[nCols];
            data = new double[nCols * nRows];
            samples = new string[nRows];
            labels = new double[nLabels * nRows];
            labelNames = new string[nLabels];
            
            // get name of sample from file name
            vector<string> folders = split(path, '/');
            string fName = folders.back();
            samples[0] = fName.substr(0, fName.size() - 4);
            
            // set labels to -1 since unknown
            for(int l = 0; l < nLabels; l++)
                labels[l] = -1;
            
            // set header values
            for(int c = 0; c < nCols; c++)
                header[c] = interpolateTo->header[c];
            
            // set label names
            for(int l = 0; l < nLabels; l++)
                labelNames[l] = interpolateTo->labelNames[l];
            
            // set data values
            if(minW > interpolateTo->header[0] || maxW < interpolateTo->header[nCols - 1]) {
                cout << "error spectrum at path cannot fit to interpolateTo DataSet, check min and max values" << endl;
                return;
            }
            int currIdx = 0;
            double lastW = -1.0;
            for(map<double, double>::iterator it = values.begin(); it != values.end(); it++) {
                if(it->first >= interpolateTo->header[currIdx]) {
                    if(it->first == interpolateTo->header[currIdx])
                        data[currIdx] = it->second;
                    else {
                        double rise = it->second - values[lastW];
                        double run = it->first - lastW;
                        double deltaX = interpolateTo->header[currIdx] - lastW;
                        data[currIdx] = lastW + deltaX * rise / run;
                    }
                    currIdx++;
                }
                lastW = it->first;
            }
        }
        
        // read .csv file of multiple spectra generated from the nasa people
        else if(path.substr(path.size() - 4) == ".csv") {
            ifstream in(path);
            string line = "";
            getline(in, line);
            vector<string> tokens = split(line, ',');
            vector<double> headerVals;
            vector<string> labelHNames;
            int i = 1;
            for (; i < tokens.size(); i++) {
                if(!isDigit(tokens.at(i).at(0)))
                    break;
                headerVals.push_back(stod(tokens.at(i)));
            }
            for (; i < tokens.size(); i++)
                labelHNames.push_back(tokens.at(i));
            nCols = (int)headerVals.size();
            header = new double[nCols];
            for (int j = 0; j < nCols; j++)
                header[j] = headerVals.at(j);
            nLabels = (int)labelHNames.size();
            labelNames = new string[nLabels];
            for (int j = 0; j < nLabels; j++)
                labelNames[j] = labelHNames.at(j);
            vector< vector<double> > hData;
            vector< vector<double> > hLabels;
            vector<string> hSamples;
            while (getline(in, line)) {
                tokens = split(line, ',');
                hSamples.push_back(tokens.at(0));
                vector<double> dataRow;
                vector<double> labelsRow;
                i = 1;
                for (int j = 0; i < tokens.size() && i <= nCols; i++, j++)
                    dataRow.push_back(stod(tokens.at(i)));
                for (int j = 0; i < tokens.size(); i++, j++)
                    labelsRow.push_back(stod(tokens.at(i)));
                hData.push_back(dataRow);
                hLabels.push_back(labelsRow);
            }
            nRows = (int)hData.size();
            data = new double[nCols * nRows];
            for (int r = 0; r < nRows; r++)
                for (int c = 0; c < nCols; c++)
                    data[r * nCols + c] = hData.at(r).at(c);
            samples = new string[nRows];
            for (int r = 0; r < nRows; r++)
                samples[r] = hSamples.at(r);
            nLabels = (int)hLabels.front().size();
            labels = new double[nRows * nLabels];
            for (int r = 0; r < nRows; r++)
                for (int c = 0; c < nLabels; c++)
                    labels[r * nLabels + c] = hLabels.at(r).at(c);
        }

        cout << "file read with " << nRows << " rows, " << nCols << " data columns, and " << nLabels << " labels" << endl;
    }
};

/*
 class to handle activation functions
 */
class ActFunc {
public:
    virtual double out(double in) = 0;
    virtual double outPrime(double in) = 0;
    virtual string toString() = 0;
};

/*
 Exponential Linear Unit used for continous outputs
 */
class ELU : public ActFunc {
private:
    static ELU* singleton;
    ELU() {
        singleton = this;
    };
public:
    double alpha = 1.0;
    static ELU* get() {
        if(singleton == NULL)
            new ELU();
        return singleton;
    };
    double out(double in) {
        if(in > 0)
            return in;
        return alpha * (exp(in) - 1.0);
    };
    double outPrime(double in){
        if(in > 0)
            return 1.0;
        return alpha * exp(in);
    };
    string toString() {
        return "ELU";
    };
};
ELU* ELU::singleton = NULL;

/*
 Sigmoid used for probabilities of features being active or of being from a class
 */
class SIG : public ActFunc {
private:
    static SIG* singleton;
    SIG() {
        singleton = this;
    };
public:
    static SIG* get() {
        if(singleton == NULL)
            new SIG();
        return singleton;
    };
    double out(double in) {
        return 1.0 / (1.0 + exp(-1.0 * in));
    };
    double outPrime(double in){
        double sig = out(in);
        return sig * (1 - sig);
    };
    string toString() {
        return "SIG";
    };
};
SIG* SIG::singleton = NULL;

/*
 Multi Layer Perceptron class
 user needs to input layers into constructor (vector where each element is number of nodes in that layer)
 MLP takes in DataSet objects for training and predictions
 All add-ons are turned off by default, you can either change default values or edit them after creating an MLP object (they are all public except for dropOut vars which you must call the setDropOut() method to use)
 */
class MLP {
private:
    bool dropOut = false;
    double visDOProb = 1.0;
    double hidDOProb = 1.0;
public:
    // nodes to keep track of activation function values
    int nNodes = 0;
    int nLayers = 0;
    
    // activation functions for each node
    ActFunc*** actFuncs = NULL;
    
    // bias terms for activation function
    double** bias = NULL;
    
    // last change in bias
    double** dBias = NULL;
    
    // weights to filter input layer
    double** weights = NULL;
    
    // last change in weights
    double** dWeights = NULL;
    
    // number of nodes per layer
    vector<int> layers;
    
    // batch size to update weights by taking average gradient over multiple training vectors
    int batchSize = 1; // 1 to not use mini-batches
    
    // learning rate, controlling small steps on how quickly the weights and bias terms update
    double learningRate = 1.0; // 1 to not use
    
    // momentum, controlling averaged time-based steps in weights and bias terms updates
    double learningMomentum = 0.0; // 0 to not use
    
    // regularization for weight decay, adds or subtracts a weighted value to approach zero
    double L1 = 0.0; // 0 to not use
    double L2 = 0.0; // 0 to not use
    
    // drop out to randomly turn nodes on and off during training, then take weighted average for prediction
    void setDropOut(bool dropOut, double visDOProb, double hidDOProb) {
        this->dropOut = dropOut;
        this->visDOProb = visDOProb;
        this->hidDOProb = hidDOProb;
        // if not using DO, then set dropped and probabilities to 1 so RBM uses all nodes equally
        if (!dropOut) {
            for(int l = 0; l < layers.size(); l++)
                for (int n = 0; n < layers.at(l); n++)
                    dropped[l][n] = 1.0;
            visDOProb = 1.0;
            hidDOProb = 1.0;
        }
    }
    double** dropped = NULL; // keep track of which nodes are turned on during drop out
    
    // cyclic learning rate/momentum cycles through multiple learning params to better escape saddle points in the loss function
    bool cyclicLearning = false;
    double rateMin = 0.01;
    double rateMax = 0.1;
    int rateStepSize = 4; // number of epochs to move from max to min, the "ideal" number is actually adaptive depedning on number of training rows (see train method, it is commented out)
    int iteration = 0; // keep track of current training iteration to cycle through learning rate/momentum, you can set this at anytime to adjust the phase of the learning cycle
    
    // max-norm to restrict weights to an n-dimensional sphere with given radius so that weights do not blow up
    bool maxNorm = false;
    double radiusLength = 2.0;
    
    // keep track of training/prediction time
    long trainingTime = 0;
    long trainingCounter = 0;
    long predictionTime = 0;
    long predictionCounter = 0;
    
    /*
     sets a percent of activation funcitons in a layer
     changes weights and bias terms to reflect activation function
     */
    void setActFuncs(int layer, ActFunc* actFunc) {
        ActFunc** l_actFuncs = actFuncs[layer];
        for(int n = 0; n < layers.at(layer); n++)
            l_actFuncs[n] = actFunc;
    }
    
    /*
     normalize labels, and save values
     */
    double a = 0.01;
    double b = 0.99;
    double* minL = NULL;
    double* maxL = NULL;
    void normalize(DataSet* set) {
        // find minL and maxL if first time calling function
        if(minL == NULL) {
            minL = new double[set->nLabels];
            maxL = new double[set->nLabels];
            
            for(int l = 0; l < set->nLabels; l++) {
                minL[l] = DBL_MAX;
                maxL[l] = 0.0;
                for(int r = 0; r < set->nRows; r++) {
                    minL[l] = min(set->labels[r * set->nLabels + l], minL[l]);
                    maxL[l] = max(set->labels[r * set->nLabels + l], maxL[l]);
                }
                for(int r = 0; r < set->nRows; r++)
                    set->labels[r * set->nLabels + l] = a + (b - a) * (set->labels[r * set->nLabels + l] - minL[l]) / (maxL[l] - minL[l]);
            }
        }
        // otherwise normalize to previously found values
        else
            for(int l = 0; l < set->nLabels; l++)
                for(int r = 0; r < set->nRows; r++)
                    set->labels[r * set->nLabels + l] = a + (b - a) * (set->labels[r * set->nLabels + l] - minL[l]) / (maxL[l] - minL[l]);
    }
    void unnormalize(DataSet* set) {
        for(int l = 0; l < set->nLabels; l++)
            for(int r = 0; r < set->nRows; r++) {
                set->predictions[r * set->nLabels + l] = minL[l] + ( (maxL[l] - minL[l]) * (set->predictions[r * set->nLabels + l] - a) ) / (b - a);
                set->labels[r * set->nLabels + l] = minL[l] + ( (maxL[l] - minL[l]) * (set->labels[r * set->nLabels + l] - a) ) / (b - a);
            }
    }
    
    /*
     destructor, clear up mem
     */
    ~MLP() {
        
        // delete all arrays of doubles
        for (int l = 0; l < nLayers; l++) {
            delete[] dropped[l];
            delete[] bias[l];
            delete[] dBias[l];
            delete[] weights[l];
            delete[] dWeights[l];
            delete[] actFuncs[l];
        }
        
        // delete pointers to arrays
        delete[] dropped;
        delete[] bias;
        delete[] dBias;
        delete[] weights;
        delete[] dWeights;
        delete[] actFuncs;
    }
    /*
     returns a string hinting towards MLP hyper parameters, used as a unique name for hyper parameters
     */
    string getName() {
        ostringstream name;
        name << scientific << setprecision(1);
        name << "MLP";
        name << "_L";
        for (int l = 0; l < nLayers; l++) {
            if (l > 0)
                name << ";";
            if (l > 0)
                name << actFuncs[l][0]->toString();
            name << layers.at(l);
        }
        name << "_B" << batchSize;
        name << "_C";
        if (!cyclicLearning)
            name << "0";
        else {
            name << "1";
            name << ";" << rateMin;
            name << ";" << rateMax;
            name << ";" << rateStepSize;
        }
        if (L1 != 0)
            name << "_L1" << L1;
        else
            name << "_L10";
        if (L2 != 0)
            name << "_L2" << L2;
        else
            name << "_L20";
        name << "_M";
        if (!maxNorm)
            name << "0";
        else {
            name << "1";
            name << "&" << radiusLength;
        }
        name << "_D";
        if (!dropOut)
            name << "0";
        else {
            name << "1";
            name << ";" << visDOProb;
            name << ";" << hidDOProb;
        }
        return name.str();
    }
    
    /*
     write MLP out with hyper-parameters, trained parameters, and memory/time benchmarks
     only parameter is file path (without extension) to write .csv
     */
    void write(string path) {
        ofstream out(path + ".csv");
        
        // output training hyper-parameters
        out << "Multi-Layer Perceptron parameters...\n";
        out << "Layers:";
        for (int i = 0; i < layers.size(); i++)
            out << "," << layers.at(i);
        out << "\n";
        out << "Label_Normalization:";
        out << "," << a << "," << b;
        for(int l = 0; l < layers.back(); l++)
            out << "," << minL[l] << "," << maxL[l];
        out << "\n";
        out << "Batch_Size:," << batchSize << "\n";
        out << "Cyclic_Learning?," << (cyclicLearning ? "yes,Rate_Min:," + to_string(rateMin) + ",Rate_Max:," + to_string(rateMax) + ",Rate_Step_Size:," + to_string(rateStepSize) : "no") << "\n";
        if (!cyclicLearning) {
            out << "Learning_Rate:," << learningRate << "\n";
            out << "Learning_Momentum:," << learningMomentum << "\n";
        }
        out << "L1_Regularization:," << L1 << "\n";
        out << "L2_Regularization:," << L2 << "\n";
        out << "Max-Norm?," << (maxNorm ? "yes,Radius_Length:," + to_string(radiusLength) : "no") << "\n";
        out << "Dropout?," << (dropOut ? "yes,Input_Dropout_Probability:," + to_string(visDOProb) + ",Hidden_Dropout_Probability:," + to_string(hidDOProb) : "no") << "\n";
        if (trainingCounter > 0)
            out << "Average_Training_Time_Per_Batch:," << trainingTime / trainingCounter << ",milliseconds\n";
        if (predictionCounter > 0)
            out << "Average_Prediction_Time_Per_Vector:," << predictionTime / predictionCounter << ",milliseconds\n";
        double memory = sizeof(MLP);
        memory += 7.0 * (double)nNodes * sizeof(double);// 1d arrays with doubles used for each node (this is <= nNodes)
        memory += 3.0 * (double)nNodes * nNodes * sizeof(double); // 2d arrays with doubles used for connections between nodes (this is <= nNodes^2)
        memory += (double)nLayers * sizeof(int); // an int for how many nodes are in each layer
        out << "Estimated_Memory_Consumption_of_MLP:," << memory << ",bytes\n";
        
        // output trained parameters
        out << "\nBias Terms...\n";
        for (int l = 1; l < nLayers; l++) {
            // write what layer we are on
            if (l == nLayers - 1)
                out << "Output Layer\n";
            else
                out << "Hidden Layer" << l << "\n";
            double* l_bias = bias[l];
            for (int n = 0; n < layers.at(l); n++)
                out << l_bias[n] << ",";
            out << "\n";
        }
        out << "\nWeights...\n";
        for (int l = 1; l < nLayers; l++) {
            // write which layer we are on
            if (l == nLayers - 1)
                out << "Output Layer\n";
            else
                out << "Hidden Layer" << l << "\n";
            
            double* l_weights = weights[l];
            ActFunc** l_actFuncts = actFuncs[l];
            for (int n = 0; n < layers.at(l); n++) {
                out << l_actFuncts[n]->toString();
                for (int n0 = 0; n0 < layers.at(l - 1); n0++)
                    out << "," << l_weights[n * layers.at(l - 1) + n0];
                out << "\n";
            }
        }
        
        out.close();
    }
    
    
    /*
     read all MLP attributes (just the reverse of write)
     parameter is name of file (without file extension, assumes it is a .csv as that is what is used for writing)
     */
    static MLP* read(string path) {
        ifstream in(path + ".csv");
        string line = "";
        vector<string> tokens;
        
        // read header
        getline(in, line);
        
        // layers
        getline(in, line);
        tokens = split(line, ',');
        vector<int> layers;
        for (int i = 1; i < tokens.size(); i++)
            layers.push_back(stoi(tokens.at(i)));
        
        MLP* mlp = new MLP(layers);
        
        // label normalization
        getline(in, line);
        tokens = split(line, ',');
        mlp->a = stod(tokens.at(1));
        mlp->b = stod(tokens.at(2));
        vector<double> mins;
        vector<double> maxs;
        for(int i = 3; i < tokens.size(); i+=2) {
            mins.push_back(stod(tokens.at(i)));
            maxs.push_back(stod(tokens.at(i+1)));
        }
        mlp->minL = new double[mins.size()];
        mlp->maxL = new double[maxs.size()];
        for(int i = 0; i < mins.size(); i++) {
            mlp->minL[i] = mins.at(i);
            mlp->maxL[i] = maxs.at(i);
        }
        
        // batches
        getline(in, line);
        tokens = split(line, ',');
        mlp->batchSize = stoi(tokens.at(1));
        
        // cyclic learning
        getline(in, line);
        if (line.substr(0, 19) == "Cyclic_Learning?,no") {
            mlp->cyclicLearning = false;
            getline(in, line);
            tokens = split(line, ',');
            mlp->learningRate = stod(tokens.at(1));
            getline(in, line);
            tokens = split(line, ',');
            mlp->learningMomentum = stod(tokens.at(1));
        }
        else {
            mlp->cyclicLearning = true;
            tokens = split(line, ',');
            mlp->rateMin = stod(tokens.at(3));
            mlp->rateMax = stod(tokens.at(5));
            mlp->rateStepSize = stod(tokens.at(7));
        }
        
        // L1 regularization
        getline(in, line);
        tokens = split(line, ',');
        mlp->L1 = stod(tokens.at(1));
        
        // L2 regularization
        getline(in, line);
        tokens = split(line, ',');
        mlp->L2 = stod(tokens.at(1));
        
        // max-norm
        getline(in, line);
        if (line.substr(0, 12) == "Max-Norm?,no") {
            mlp->maxNorm = false;
        }
        else {
            mlp->maxNorm = true;
            tokens = split(line, ',');
            mlp->radiusLength = stod(tokens.at(3));
        }
        
        // dropout
        getline(in, line);
        if (line.substr(0, 11) == "Dropout?,no") {
            mlp->dropOut = false;
        }
        else {
            mlp->dropOut = true;
            tokens = split(line, ',');
            mlp->visDOProb = stod(tokens.at(3));
            mlp->hidDOProb = stod(tokens.at(5));
        }
        
        // time benchmarks
        getline(in, line);
        tokens = split(line, ',');
        if (tokens.at(0) == "Average_Training_Time_Per_Batch:") {
            mlp->trainingCounter = 1.0;
            mlp->trainingTime = stod(tokens.at(1));
            getline(in, line);
            tokens = split(line, ',');
        }
        if (tokens.at(0) == "Average_Prediction_Time_Per_Vector:") {
            mlp->predictionCounter = 1.0;
            mlp->predictionTime = stod(tokens.at(1));
            getline(in, line);
        }
        getline(in, line);
        getline(in, line);
        getline(in, line);
        getline(in, line);

        int layer = 1;
        while (line.size() > 2) {
            if (line.at(0) == 'O' || line.at(0) == 'H') {
                getline(in, line);
                layer++;
            }
            else {
                tokens = split(line, ',');
                for (int i = 0; i < tokens.size(); i++)
                    mlp->bias[layer][i] = stod(tokens.at(i));
                getline(in, line);
            }
        }
        getline(in, line);
        getline(in, line);

        layer = 1;
        int i = 0;
        int n = 0;
        while (getline(in, line)) {
            if (line.at(0) == 'O' || line.at(0) == 'H') {
                layer++;
                i = 0;
                n = 0;
            }
            else {
                tokens = split(line, ',');
                string actFunc = tokens.front();
                if (actFunc == "ELU")
                    mlp->actFuncs[layer][n++] = ELU::get();
                for (int n0 = 1; n0 < tokens.size(); n0++)
                    mlp->weights[layer][i++] = stod(tokens.at(n0));
            }
        }

        
        in.close();
        return mlp;
    }
    
    /*
     create MLP with specified layers, initialize weights and other needed variables
     */
    MLP(vector<int> layers) {
        this->layers = layers;
        
        // create nodes that will store activation function values
        nNodes = 0;
        nLayers = (int)layers.size();
        
        // create our arrays of numbers needed for each layer
        // (some layers do not need these numbers but I kept dummy values there to maintain a simple layer index architecture). A more complicated one of just one matrix of values is overkill as the number of math operations to get the proper index isn't worth the trade off in time when accessing an element in a 1d array
        actFuncs = new ActFunc**[nLayers];
        dropped = new double*[nLayers];
        bias = new double*[nLayers];
        dBias = new double*[nLayers];
        weights = new double*[nLayers];
        dWeights = new double*[nLayers];
        
        // create dummy weights to keep proper layer index architecture
        weights[0] = new double[1];
        weights[0][0] = 0.0;
        dWeights[0] = new double[1];
        dWeights[0][0] = 0.0;
        
        // create normal distributed RNG around 0 with a variance close to zero to randomly initialize some of the parameters
        normal_distribution<double> normallyDistributed(0.0, 1.0 / sqrt((double(layers.front() + layers.back()))));
        default_random_engine rng;
        
        // create layer arrays and initialize values in arrays
        for (int l = 0; l < nLayers; l++) {
            nNodes += layers.at(l);
            
            // create layer arrays
            actFuncs[l] = new ActFunc*[layers.at(l)];
            dropped[l] = new double[layers.at(l)];
            bias[l] = new double[layers.at(l)];
            dBias[l] = new double[layers.at(l)];
            
            // create 2d arrays for this layer
            if (l != 0) {
                weights[l] = new double[layers.at(l) * layers.at(l - 1)];
                dWeights[l] = new double[layers.at(l) * layers.at(l - 1)];
            }
            
            // move through each node in this layer
            for (int n = 0; n < layers.at(l); n++) {
                
                // set 1d array values for this layer
                actFuncs[l][n] = SIG::get();
                dropped[l][n] = 1.0;
                bias[l][n] = normallyDistributed(rng);
                dBias[l][n] = 0.0;
                
                // 2d arrays...
                if (l != 0) {
                    // move through each node in previous layer to build any links between nodes
                    for (int n0 = 0; n0 < layers.at(l - 1); n0++) {
                        
                        // set 2d array values for these connected layers
                        weights[l][n * layers.at(l - 1) + n0] = normallyDistributed(rng);
                        dWeights[l][n * layers.at(l - 1) + n0] = 0.0;
                    }
                }
            }
        }
    }
    
    // forward propagate through layers and save predictions to data set, deletes int[] rows
    void predictThread(DataSet* set, int* rows, int nRows) {
        // since this is capable of multithreading, we need to create vars we can locally write to
        double** nodes = new double*[layers.size()];
        for (int l = 0; l < layers.size(); l++) {
            nodes[l] = new double[layers.at(l)];
            for(int n = 0; n < layers.at(l); n++)
                nodes[l][n] = 0.0;
        }
        
        for(int rI = 0; rI < nRows; rI++) {
            
            int r = rI;
            if(rows)
                r = rows[rI];
            
            for (int l = 0; l < layers.size(); l++) {
                
                // grab layer of nodes
                double* l_nodes = nodes[l]; // read and write
                
                // input data into the first layer
                if (l == 0) {
                    for (int n = 0; n < layers.front(); n++)
                        l_nodes[n] = set->data[r * set->nCols + n];
                    continue;
                }
                
                double* l0_nodes = nodes[l - 1]; // read only
                double* l_bias = bias[l]; // read only
                double* l0_dropped = dropped[l - 1]; // read only
                double* l_weights = weights[l]; // read only
                ActFunc** l_actFuncs = actFuncs[l]; // call
                
                for (int n = 0; n < layers.at(l); n++) {
                    
                    // sum a response from bias plus weighted inputs
                    double sum = l_bias[n];
                    for (int n0 = 0; n0 < layers.at(l - 1); n0++)
                        sum += l_weights[n * layers.at(l - 1) + n0] * l0_nodes[n0] * l0_dropped[n0];
                    
                    // input sum into the activation function (in this example, sigmoid)
                    double actFun = l_actFuncs[n]->out(sum);
                    
                    // save output of activation function to node value
                    l_nodes[n] = actFun;
                    
                    // save predictions to dataset
                    if(l == nLayers - 1)
                        set->predictions[r * set->nLabels + n] = actFun;
                }
            }
        }
        
        // clear mem
        for (int l = 0; l < nLayers; l++)
            delete[] nodes[l];
        delete[] nodes;
        delete[] rows;
    }
    
    // forward and backward propagate through layers while maintaing own data, deletes int[] rows
    void trainThread(DataSet* set, int* rows, int nRows, double** gWeights, double** gBias) {
        // create local variables used in propagation
        double** nodes = new double*[layers.size()];
        double** derivatives = new double*[layers.size()];
        double** backProp = new double*[layers.size()];
        for (int l = 0; l < layers.size(); l++) {
            nodes[l] = new double[layers.at(l)];
            derivatives[l] = new double[layers.at(l)];
            backProp[l] = new double[layers.at(l)];
            for(int n = 0; n < layers.at(l); n++) {
                nodes[l][n] = 0.0;
                derivatives[l][n] = 0.0;
                backProp[l][n] = 0.0;
            }
        }
        
        for(int rI = 0; rI < nRows; rI++) {
            
            int r = rows[rI];
            
            // forward prop
            for (int l = 0; l < layers.size(); l++) {
                
                // grab layer of nodes
                double* l_nodes = nodes[l]; // read and write
                
                // input data into the first layer
                if (l == 0) {
                    for (int n = 0; n < layers.front(); n++)
                        l_nodes[n] = set->data[r * set->nCols + n];
                    continue;
                }
                
                double* l0_nodes = nodes[l - 1]; // read only
                double* l_bias = bias[l]; // read only
                double* l_actFuncDerivs = derivatives[l]; // read and write
                double* l0_dropped = dropped[l - 1]; // read only
                double* l_weights = weights[l]; // read only
                ActFunc** l_actFuncs = actFuncs[l]; // call
                
                for (int n = 0; n < layers.at(l); n++) {
                    
                    // sum a response from bias plus weighted inputs
                    double sum = l_bias[n];
                    for (int n0 = 0; n0 < layers.at(l - 1); n0++)
                        sum += l_weights[n * layers.at(l - 1) + n0] * l0_nodes[n0] * l0_dropped[n0];
                    
                    // input sum into the activation function (in this example, sigmoid)
                    double actFun = l_actFuncs[n]->out(sum);
                    
                    // save output of activation function to node value
                    l_nodes[n] = actFun;
                    
                    // calculate derivative of activation function and save to node derivative value
                    l_actFuncDerivs[n] = l_actFuncs[n]->outPrime(sum);
                }
            }
            
            // now back prop
            for (int l = nLayers - 1; l > 0; l--) {
                double* l_nodes = nodes[l]; // read and write
                double* l_actFuncDerivs = derivatives[l]; // read and write
                double* l_gWeights = gWeights[l]; // read and write
                double* l_weights = weights[l]; // read only
                double* l_gBias = gBias[l]; // read and write
                double* l_backProp = backProp[l]; // read and write
                double* l0_backProp = backProp[l - 1]; // read and write
                double* l0_dropped = dropped[l - 1]; // read only
                double* l_dropped = dropped[l]; // read only
                double* l0_nodes = nodes[l - 1]; // read and write
                
                for (int n = 0; n < layers.at(l); n++) {
                    
                    // derivative of error with respect ot this node
                    double dError_dNode = l_backProp[n];
                    l_backProp[n] = 0.0; // reset for next training vector
                    
                    if(l_dropped[n] == 0)
                        continue;
                    
                    // output layer
                    if (l == nLayers - 1) {
                        // get the expected value for this output node
                        double expected = set->labels[r * set->nLabels + n];
                        
                        // calculate the error, no need to really calculate this, it is just to show you
                        // double error = 0.5 * pow(nodes[nNodesb4Output + b] - set->labels[r * set->nLabels + b], 2);
                        
                        // derivative of error with respect to output node
                        dError_dNode = expected - l_nodes[n]; // 2 * 0.5 (output - expected) * -1
                    }
                    
                    // derivative of output node with respect to weighted sum
                    double dNode_dSum = l_actFuncDerivs[n]; // derivative of activation function of weighted sum
                    
                    // derivative of weighted sum with respect to bias term
                    double dSum_dBias = 1.0; // partial derivative (respect to bias) of -> bias + weight_0 * value_0 + ... + weight_j * value_j + ... + weight_n * value_n = 1;
                    
                    // use chain rule to calculate bias gradient for this output node
                    double dError_dBias = dError_dNode * dNode_dSum * dSum_dBias;
                    
                    // add error gradient of bias to sum over this batch (will be averaged later)
                    l_gBias[n] += dError_dBias; // this is shared by all threads, use mutex
                    
                    // calculate weight gradients
                    for (int n0 = 0; n0 < layers.at(l - 1); n0++) {
                        if (l0_dropped[n0] == 0)
                            continue;
                        
                        // derivative of weighted sum with respect to weight connecting out to hid node
                        double dSum_dWeight = l0_nodes[n0]; // partial derivative (respect to weight_n) of -> bias + weight_0 * value_0 + ... + weight_n * value_n + ... + weight_N * value_N = value_n
                        
                        // chain rule to find derivative of error with respect to this weight
                        double dError_dWeight = dError_dNode * dNode_dSum * dSum_dWeight; // chain rule
                        
                        // add error gradient to sum over this batch (to be averaged after batch)
                        l_gWeights[n * layers.at(l - 1) + n0] += dError_dWeight; // this is shared by all threads, use mutex
                        
                        // calculate back prop error gradient for previous layer
                        if (l > 1) {
                            // derivative of weighted sum with respect to hidden node
                            double dSum_dHid = l_weights[n * layers.at(l - 1) + n0]; // partial derivative (respect to value_n) of hidden node -> bias + weight_0 * value_0 + ... + weight_n * value_n + ... + weight_N * value_N = weight_n
                            
                            // chain rule to find derivative of error with respect to hidden node
                            double dError_dHid = dError_dNode * dNode_dSum * dSum_dHid;
                            
                            // add to back propagation gradient for this hidden node
                            l0_backProp[n0] += dError_dHid;
                        }
                    }
                }
            }
        }
        
        // clear mem
        for (int l = 0; l < layers.size(); l++) {
            delete[] backProp[l];
            delete[] nodes[l];
            delete[] derivatives[l];
        }
        delete[] backProp;
        delete[] nodes;
        delete[] derivatives;
        delete[] rows;
    }
    
    /*
     train MLP for one full epoch
     takes in a data set to train on
     return Squared Percent Error of training set found during training (this can be used to update SPE but not completely accurate as it changes throughout training process)
     */
    void train(DataSet* set) {
        // time how long it takes
        StopWatch stopWatch;
        int nBatches = 0;
        
        // randomly shuffle indices for training set
        double* randomRows = new double[set->nRows];
        for (int i = 0; i < set->nRows; i++)
            randomRows[i] = i;
        for (int i = 0; i < set->nRows; i++) {
            int randRow = rand() % set->nRows;
            if(i != randRow) {
                int temp = randomRows[i];
                randomRows[i] = randomRows[randRow];
                randomRows[randRow] = temp;
            }
        }
        // start our random row index at 0, and increment as we train to grab a new random row
        int randRowIdx = 0;
        
        // create arrays to keep track of parameter gradientds
        double** gWeights = new double*[nLayers];
        double** gBias = new double*[nLayers];
        for(int l = 0; l < nLayers; l++) {
            gBias[l] = new double[layers.at(l)];
            for(int n = 0; n < layers.at(l); n++)
                gBias[l][n] = 0.0;
            if(l == 0) {
                gWeights[0] = new double[1];
                gWeights[0][0] = 0.0;
            }
            else {
                gWeights[l] = new double[layers.at(l) * layers.at(l-1)];
                for(int n = 0; n < layers.at(l); n++)
                    for(int n0 = 0; n0 < layers.at(l-1); n0++)
                        gWeights[l][n * layers.at(l-1) + n0] = 0.0;
            }
        }
        
        // make some multithread variables, they do not get used if user is not multithreading
        int nLocalThreads = 0;
        int nThreadRows = 0;
        double*** tGWeights = NULL;
        double*** tGBias = NULL;
        thread* threads = NULL;
        bool doMultiThread = canMultiThread && batchSize > 1;
        if(doMultiThread) {
            
            // get number of threads we will use for each mini-batch
            nLocalThreads = min(nGlobalThreads, batchSize);
            
            // get number of batch rows per thread, the last thread may be less than this if not even (why we add one if mod has a value)
            nThreadRows = max(1, batchSize / nLocalThreads);
            if(batchSize % nLocalThreads > 0)
                nThreadRows++;
            
            // make array to keep track of threads
            threads = new thread[nLocalThreads];
            
            // make arrays to keep track of parameter gradients for each thread (seperate data to write to)
            tGWeights = new double**[nLocalThreads];
            tGBias = new double**[nLocalThreads];
            for(int t = 0; t < nLocalThreads; t++) {
                tGWeights[t] = new double*[nLayers];
                tGBias[t] = new double*[nLayers];
                for(int l = 0; l < nLayers; l++) {
                    tGBias[t][l] = new double[layers.at(l)];
                    for(int n = 0; n < layers.at(l); n++)
                        tGBias[t][l][n] = 0.0;
                    if(l == 0) {
                        tGWeights[t][0] = new double[1]; // dummy value to keep architechture of arrays
                        tGWeights[t][0][0] = 0.0;
                    }
                    else {
                        tGWeights[t][l] = new double[layers.at(l) * layers.at(l-1)];
                        for(int n = 0; n < layers.at(l); n++)
                            for(int n0 = 0; n0 < layers.at(l-1); n0++)
                                tGWeights[t][l][n*layers.at(l-1) + n0] = 0.0;
                    }
                }
            }
        }
        
        // move through each row in training set, splitting up into mini-batches if batchSize > 1
        while(randRowIdx < set->nRows) {
            nBatches++;
            
            // if using dropout, then randomly drop some nodes for this batch
            if (dropOut) {
                // get random numbers for this batch to randomly drop some visible nodes
                for (int l = 0; l < layers.size() - 1; l++) {
                    double* lDropped = dropped[l];
                    for (int n = 0; n < layers.at(l); n++) {
                        // get a uniformly random number between [0,1]
                        double ran = ((double)rand() / (RAND_MAX));
                        
                        // randomly drop this node
                        double prob = hidDOProb;
                        if (l == 0)
                            prob = visDOProb;
                        if (prob >= ran)
                            lDropped[n] = 1.0;
                        else
                            lDropped[n] = 0.0;
                    }
                }
            }
            
            // split into batches, if batchSize == 1 then this will mathematically not do any batches but logic is same
            int nBatchRows = 0;
            if(doMultiThread) {
                
                // take a set of training rows from this batch and send to their own thread
                int t = 0;
                while(nBatchRows < batchSize && randRowIdx < set->nRows) {
                    int idx = 0;
                    int* threadRows = new int[nThreadRows];
                    for(; idx < nThreadRows && nBatchRows < batchSize && randRowIdx < set->nRows; nBatchRows++)
                        threadRows[idx++] = randomRows[randRowIdx++];
                    threads[t] = thread(&MLP::trainThread, this, set, threadRows, idx, tGWeights[t], tGBias[t]);
                    t++;
                }
                
                // wait for threads to finish
                for(int i = 0; i < t; i++)
                    threads[i].join();
                
                // add up all the multi-threaded gradients to update parameters, and reset the multi-threaded gradients to zero for next batch
                for(t = 0; t < nLocalThreads; t++) {
                    double** t_gWeights = tGWeights[t];
                    double** t_gBias = tGBias[t];
                    for (int l = 1; l < layers.size(); l++) {
                        double* l_gWeights = gWeights[l];
                        double* l_gBias = gBias[l];
                        double* l_tGWeights = t_gWeights[l];
                        double* l_tGBias = t_gBias[l];
                        for (int n = 0; n < layers.at(l); n++) {
                            l_gBias[n] += l_tGBias[n];
                            l_tGBias[n] = 0.0;
                            for (int n0 = 0; n0 < layers.at(l-1); n0++) {
                                l_gWeights[n * layers.at(l-1) + n0] += l_tGWeights[n * layers.at(l-1) + n0];
                                l_tGWeights[n * layers.at(l-1) + n0] = 0.0;
                            }
                        }
                    }
                }
            }
            else {
                int* batchRows = new int[batchSize];
                for(; nBatchRows < batchSize && randRowIdx < set->nRows; nBatchRows++)
                    batchRows[nBatchRows] = randomRows[randRowIdx++];
                trainThread(set, batchRows, nBatchRows, gWeights, gBias);
            }
            
            // determine learning rate (either stagnant or cyclic)
            double thisLearningRate = learningRate;
            double thisLearningMomentum = learningMomentum;
            if (cyclicLearning) {
                // rateStepSize = 2.0 * (double)set->nRows / (double)batchSize; // "ideal" adaptive step size
                // cycles through using trianglar learning policy /\/\/\/\
                // each slash takes the stepSize number of epochs to go from bottom to top (or vice versa)
                thisLearningRate = rateMin + ((iteration / (rateStepSize - 1)) % 2 == 1 ? rateStepSize - 1 - iteration % (rateStepSize - 1) : iteration % (rateStepSize - 1)) * (rateMax - rateMin) / ((double)(rateStepSize - 1));
            }
            
            // now that back propagation is done for this batch of training vectors, we need to update our parameters...
            for (int l = 1; l < layers.size(); l++) {
                double* l_dBias = dBias[l];
                double* l_gBias = gBias[l];
                double* l_bias = bias[l];
                double* l_dWeights = dWeights[l];
                double* l_gWeights = gWeights[l];
                double* l_weights = weights[l];
                double* l_dropped = dropped[l];
                double* l0_dropped = dropped[l-1];
                
                for (int n = 0; n < layers.at(l); n++) {
                    if (l_dropped[n] == 0)
                        continue;
                    
                    // calculate the change in bias term
                    l_dBias[n] *= thisLearningMomentum;
                    l_dBias[n] += thisLearningRate * l_gBias[n] / ((double)nBatchRows);
                    l_dBias[n] += thisLearningRate * L1 * (l_bias[n] == 0 ? 0.0 : (l_bias[n] > 0 ? -1.0 : 1.0));
                    
                    // update bias term for first time
                    l_bias[n] += l_dBias[n];
                    
                    // L2 update (do this after to insure L2 is always pushing the parameters towards zero
                    l_dBias[n] -= thisLearningRate * L2 * l_bias[n];
                    l_bias[n] -= thisLearningRate * L2 * l_bias[n];
                    l_gBias[n] = 0.0; // reset for next batch
                    
                    // update weights
                    for (int n0 = 0; n0 < layers.at(l-1); n0++) {
                        if (l0_dropped[n0] == 0)
                            continue;
                        
                        // calculate the change in weight
                        int idx = n * layers.at(l - 1) + n0;
                        l_dWeights[idx] *= thisLearningMomentum;
                        l_dWeights[idx] += thisLearningRate * l_gWeights[idx] / ((double)nBatchRows);
                        l_dWeights[idx] += thisLearningRate * L1 * (l_weights[idx] == 0 ? 0.0 : (l_weights[idx] > 0 ? -1.0 : 1.0));
                        
                        // update weight for first time
                        l_weights[idx] += l_dWeights[idx];
                        
                        // L2 update (do this after to insure L2 is always pushing the parameters towards zero
                        l_dWeights[idx] -= thisLearningRate * L2 * l_weights[idx];
                        l_weights[idx] -= thisLearningRate * L2 * l_weights[idx];
                        l_gWeights[idx] = 0.0; // reset for next batch
                    }
                }
            }
            
            // if using max-norm, make sure the radial length of weights (for each node, excluding the bias) fit into an n-dimensial sphere with given radius
            if (maxNorm) {
                for (int l = 1; l < layers.size(); l++) {
                    double* l_dropped = dropped[l];
                    double* l0_dropped = dropped[l - 1];
                    double* l_weights = weights[l];
                    double* l_dWeights = dWeights[l];
                    
                    for (int n = 0; n < layers.at(l); n++) {
                        if (l_dropped[n] == 0)
                            continue;
                        
                        // find length of weight radius
                        double length = 0.0;
                        for (int n0 = 0; n0 < layers.at(l - 1); n0++) {
                            if (l0_dropped[n0] == 0)
                                continue;
                            
                            length += pow(l_weights[n * layers.at(l - 1) + n0], 2);
                        }
                        length = sqrt(length);
                        
                        // adjust length to fit in weight sphere
                        if (length > radiusLength) {
                            for (int n0 = 0; n0 < layers.at(l - 1); n0++) {
                                if (l0_dropped[n0] == 0)
                                    continue;
                                
                                l_dWeights[n * layers.at(l - 1) + n0] += l_weights[n * layers.at(l - 1) + n0] * (radiusLength / length - 1);
                                l_weights[n * layers.at(l - 1) + n0] *= radiusLength / length;
                            }
                        }
                    }
                }
            }
            
            // log that we completed one training iteration, used for cyclic learning
            iteration++;
        }
        // clean up mem
        for(int l = 0; l < nLayers; l++) {
            delete[] gBias[l];
            delete[] gWeights[l];
        }
        delete[] gWeights;
        delete[] gBias;
        delete[] randomRows;
        if(doMultiThread) {
            for(int t = 0; t < nLocalThreads; t++) {
                for(int l = 0; l < nLayers; l++) {
                    delete[] tGBias[t][l];
                    delete[] tGWeights[t][l];
                }
                delete[] tGWeights[t];
                delete[] tGBias[t];
            }
            delete[] tGWeights;
            delete[] tGBias;
            delete[] threads;
        }
        
        // log training results
        //cout << "training done in " << stopWatch.lap() << " with batches=" << (long)nBatches << endl;
        trainingTime += stopWatch.stop() / (long)nBatches;
        trainingCounter += 1.0;
    }
    
    /*
     use MLP to predict labels for given DataSet
     saves predictions in DataSet object
     */
    void predict(DataSet* set, bool doUnnormalize) {
        if(doUnnormalize)
            normalize(set);
        
        // keep track of how long it takes
        StopWatch stopWatch;
        
        // if using dropout, set dropout array to probabilities for when we forward propagate to calculate predictions
        if (dropOut) {
            for (int l = 0; l < layers.size() - 1; l++) {
                double* lDropped = dropped[l];
                for (int n = 0; n < layers.at(l); n++) {
                    double prob = hidDOProb;
                    if (l == 0)
                        prob = visDOProb;
                    lDropped[n] = prob;
                }
            }
        }
        
        // check if we need to make a new array that holds prediction values for each row in DataSet for each label
        if (!set->predictions)
            set->predictions = new double[set->nRows * layers.back()];
        
        // make some multithread variables, they do not get used if user is not multithreading
        int nLocalThreads = 0;
        int nThreadRows = 0;
        thread* threads = NULL;
        bool doMultiThread = canMultiThread && set->nRows > 1;
        if(doMultiThread) {
            nLocalThreads = min(nGlobalThreads, set->nRows);
            nThreadRows = max(1, set->nRows / nLocalThreads);
            if(set->nRows % nLocalThreads > 0)
                nThreadRows++;
            threads = new thread[nLocalThreads];
        }
        
        if(doMultiThread) {
            // assign a set of rows for each local thread
            int r = 0;
            int t = 0;
            while(r < set->nRows) {
                int* threadRows = new int[nThreadRows];
                int idx = 0;
                while(r < set->nRows && idx < nThreadRows)
                    threadRows[idx++] = r++;
                threads[t] = thread(&MLP::predictThread, this, set, threadRows, idx);
                t++;
            }
            
            // wait for threads to finish
            for(int i = 0; i < t; i++)
                threads[i].join();
        }
        else
            predictThread(set, NULL, set->nRows);
        
        // clear mem
        if(threads)
            delete[] threads;
        
        if(doUnnormalize)
            unnormalize(set);
        
        // log prediction benchmarks
        //cout << "predict done in " << stopWatch.lap() << " ms with rows=" << (long)set->nRows << endl;
        predictionCounter += 1.0;
        predictionTime += stopWatch.stop() / (long)set->nRows;
    }
};

/*
 classifies the data set based on prediction values (the label with the highest probability wins)
 classification is stored in DataSet in the string binaryClass
 method returns the accuracy, precision, recall, F1 score, and RMSE for each label using: map<label, map<name, value> >
 ground truths are 0 for classes it is not and 1 for the class it is
 it uses [-1] to log scores for all labels
 */
map<int, map<string, double> > binaryClassify(DataSet* set) {
    
    // time it
    StopWatch stopwatch;
    
    // make new binary class array if needs be
    if (!set->binaryClass)
        set->binaryClass = new string[set->nRows];
    
    // make binary classifications and log results
    map<int, map<string, int> > results; // map<labelIndex, map<name, value> > where name is "11"=truePositive, "10"=trueNegative, "01"=falsePositive, "00"=falseNegative. This will be used to get scores.
    map<int, double> SE; // squared error, // map<labelIndex, squaredError>
    for (int r = 0; r < set->nRows; r++) {
        
        // get the binary class of this vector
        string binClass = "";
        double maxProb = 0.0;
        for (int l = 0; l < set->nLabels; l++) {
            if (set->predictions[r * set->nLabels + l] > maxProb) {
                maxProb = set->predictions[r * set->nLabels + l];
                binClass = set->labelNames[l];
            }
            
            // calculate squared error between prediction and label values
            SE[l] += pow(set->predictions[r * set->nLabels + l] - set->labels[r * set->nLabels + l], 2);
            SE[-1] += pow(set->predictions[r * set->nLabels + l] - set->labels[r * set->nLabels + l], 2); // all
        }
        set->binaryClass[r] = binClass;
        
        // calculate true/false postives/negatives
        for (int l = 0; l < set->nLabels; l++) {
            bool testedPositive = false;
            if (binClass == set->labelNames[l])
                testedPositive = true;
            bool isPositive = false;
            if (set->labels[r * set->nLabels + l] == 0.99)
                isPositive = true;
            if (testedPositive && isPositive) {
                results[l]["11"]++;
                results[-1]["11"]++;
            }
            else if (testedPositive && !isPositive) {
                results[l]["10"]++;
                results[-1]["10"]++;
            }
            else if (!testedPositive && !isPositive) {
                results[l]["00"]++;
                results[-1]["00"]++;
            }
            else if (!testedPositive && isPositive) {
                results[l]["01"]++;
                results[-1]["01"]++;
            }
        }
    }
    
    // calculate accuracy, precision, recall, F1 scores, and RMSE from results
    map<int, map<string, double> > scores;
    for (int l = -1; l < set->nLabels; l++) {
        double total = results[l]["11"] + results[l]["10"] + results[l]["01"] + results[l]["00"];
        scores[l]["a"] = ((double)results[l]["11"] / ((double)results[l]["11"] + (double)results[l]["01"]));
        scores[l]["p"] = ((double)results[l]["11"]) / ((double)results[l]["11"] + (double)results[l]["10"]);
        scores[l]["r"] = ((double)results[l]["11"]) / ((double)results[l]["11"] + (double)results[l]["01"]);
        scores[l]["f"] = 2.0 * scores[l]["r"] * scores[l]["p"] / (scores[l]["r"] + scores[l]["p"]);
        scores[l]["rmse"] = sqrt(SE[l] / total );
    }
    
    //cout << "binary classification done in " << stopwatch.stop() << " ms" << endl;
    return scores;
}

/*
 multi-classification.
 the predictions from the MLP must be equal to or higher than the passed in probability to be classified as positive, if(prediction >= probability) then postive
 classifcations are stored in the DataSet as 0 (negative) or 1 (positive) for each label, when DataSet is written it will write the class names it is positive for
 ground truths are also 0 (negative) or 1 (positive) for each label
 method returns the accuracy, precision, recall, F1 score, and RMSE for each label map<label, map<name, value> >
 it uses [-1] to log scores for all labels
 */
map<int, map<string, double> > multiClassify(DataSet* set, double probability) {
    
    // time it
    StopWatch stopwatch;
    
    // make new mutli class array if needs be
    if (!set->multiClass)
        set->multiClass = new int[set->nRows * set->nLabels];
    
    // make multiple classifications and log results
    map<int, map<string, int> > results;
    map<int, double> SE; // squared error
    for (int r = 0; r < set->nRows; r++) {
        
        // check class probabilities
        for (int l = 0; l < set->nLabels; l++) {
            // make classification
            if (set->predictions[r * set->nLabels + l] >= probability)
                set->multiClass[r * set->nLabels + l] = 1;
            else
                set->multiClass[r * set->nLabels + l] = 0;
            
            // get squared error
            SE[l] += pow(set->predictions[r * set->nLabels + l] - set->labels[r * set->nLabels + l], 2);
            SE[-1] += pow(set->predictions[r * set->nLabels + l] - set->labels[r * set->nLabels + l], 2);
            
            // calculate true/false postives/negatives
            bool testedPositive = false;
            if (set->multiClass[r * set->nLabels + l] == 1)
                testedPositive = true;
            bool isPositive = false;
            if (set->labels[r * set->nLabels + l] == 0.99)
                isPositive = true;
            if (testedPositive && isPositive) {
                results[l]["11"]++;
                results[-1]["11"]++;
            }
            else if (testedPositive && !isPositive) {
                results[l]["10"]++;
                results[-1]["10"]++;
            }
            else if (!testedPositive && !isPositive) {
                results[l]["00"]++;
                results[-1]["00"]++;
            }
            else if (!testedPositive && isPositive) {
                results[l]["01"]++;
                results[-1]["01"]++;
            }
        }
    }
    
    // calculate accuracy, precision, recall, F1 scores, and RMSE
    map<int, map<string, double> > scores;
    for (int l = -1; l < set->nLabels; l++) {
        double total = results[l]["11"] + results[l]["10"] + results[l]["01"] + results[l]["00"];
        scores[l]["a"] = ((double)results[l]["11"] + (double)results[l]["00"]) / total;
        scores[l]["p"] = ((double)results[l]["11"]) / ((double)results[l]["11"] + (double)results[l]["10"]);
        scores[l]["r"] = ((double)results[l]["11"]) / ((double)results[l]["11"] + (double)results[l]["01"]);
        scores[l]["f"] = 2.0 * scores[l]["r"] * scores[l]["p"] / (scores[l]["r"] + scores[l]["p"]);
        scores[l]["rmse"] = sqrt(SE[l] / total);
    }
    
    //cout << "multi classification done in " << stopwatch.stop() << " ms" << endl;
    return scores;
}

/*
 REGRESSION
 method returns the RMSE (root mean squared error) for each label map<label, value>
 it uses [-1] to log RMSE for all labels
 */
map<int, double> regression(DataSet* set) {
    
    // time it
    StopWatch stopwatch;
    
    // make multiple classifications and log results
    map<int, double> RMSE; // squared error
    for (int r = 0; r < set->nRows; r++) {
        for (int l = 0; l < set->nLabels; l++) {
            // get squared error
            RMSE[l] += pow(set->predictions[r * set->nLabels + l] - set->labels[r * set->nLabels + l], 2);
            RMSE[-1] += pow(set->predictions[r * set->nLabels + l] - set->labels[r * set->nLabels + l], 2);
        }
    }
    
    // calculate RMSE
    RMSE[-1] = sqrt(RMSE[-1] / ((double)(set->nRows * set->nLabels)) );
    for (int l = 0; l < set->nLabels; l++)
        RMSE[l] = sqrt(RMSE[l] / ((double)set->nRows) );
    
    //cout << "regression done in " << stopwatch.stop() << " ms" << endl;
    return RMSE;
}

/*
 writes the results (scores) for each epoch to given file path as a .csv (without file extension)
 use notes for anything worth noting =P
 */
void writeClassificationResults(map<int, map<int, map<string, double> > > epochScores, string* labelNames, double nLabels, string filePath) {
    
    // time it
    StopWatch stopwatch;
    
    // open file for output
    ofstream writer(filePath + ".csv");
    
    // write header
    writer << "epoch,Total F1 Score,Total RMSE,Total Accuracy,Total Precision,Total Recall";
    for(int l = 0; l < nLabels; l++)
        writer << "," + labelNames[l] + " F1 Score," + labelNames[l] + " RMSE," + labelNames[l] + " Accuracy," + labelNames[l] + " Precision," + labelNames[l] + " Recall";
    writer << "\n";
    
    // write scores
    for(map<int, map<int, map<string, double> > >::iterator it = epochScores.begin(); it != epochScores.end(); it++) {
        writer << it->first;
        for(int l = -1; l < nLabels; l++)
            writer << "," << it->second[l]["f"] << "," << it->second[l]["rmse"] << "," << 100.0 * it->second[l]["a"] << "%," << 100.0 * it->second[l]["p"] << "%," << 100.0* it->second[l]["r"] << "%";
        writer << "\n";
    }
    
    writer.close();
    //cout << "results written in " << stopwatch.stop() << " ms to file " << filePath << endl;
}

/*
 writes the results (RMSE) for each epoch to given file path as a .csv (without file extension)
 use notes for anything worth noting =P
 */
void writeRegressionResults(map<int, map<int, double> > epochRMSPE, string* labelNames, double nLabels, string filePath) {
    
    // time it
    StopWatch stopwatch;
    
    // open file for output
    ofstream writer(filePath + ".csv");
    
    // write header
    writer << "epoch,Total RMSE";
    for(int l = 0; l < nLabels; l++)
        writer << "," + labelNames[l] + " RMSE";
    writer << "\n";
    
    // write scores
    for(map<int, map<int, double> >::iterator it = epochRMSPE.begin(); it != epochRMSPE.end(); it++) {
        writer << it->first;
        for(int l = -1; l < nLabels; l++)
            writer << "," << it->second[l];
        writer << "\n";
    }
    
    writer.close();
    //cout << "results written in " << stopwatch.stop() << " ms to file " << filePath << endl;
}

/*
    reads models from raw .txt and write to .csv files
    @param1 vector<double> bins .at(0)=start, .at(1)=end, .at(2)=increment, (empty vector to not bin)
    @param2 bool ignoreOutOfBounds=true will skipp all albedo values less than 0 and greater than 1
    @param3 string modelFolder path to folder with relectance spectra models
    @param4 string cloudFolder path to folder with cloud profiles to only reads model with cloud profiles, pass an empty string to ignore
    @param5 string outputFolder path to folder which will write models as .csv files
    @param6 is to add a tag to end of .csv master file that is written at end
*/
void extractModels(vector<double> bins, bool ignoreOutOfBounds, string modelFolder, string cloudFolder, string outputFolder, string tag) {
    
    // check if we are to bin models
    bool bin = false;
    if (bins.size() > 0)
        bin = true;

    // create list of all models with cloud profiles
    map<string, bool> clouds;
    if (!cloudFolder.empty()) {
        // get all cloud file names
        vector<string> cFiles = getFilesInFolder(cloudFolder, false);
        for (string c : cFiles) {

            // get all parts of each cloud file name (temperature, gravity, metallicity, f_sed)
            vector<string> parts = split(c, '_');
            string t = parts.at(1).substr(4, 3);
            string g = parts.at(3).substr(1);
            string m = parts.at(4).substr(1);
            string f = parts.at(5).substr(1);

            // add to clouds map to show cloud profile exists for this model
            clouds[t + g + m + f] = true;
        }
    }

    // create output folder
    createFolder(outputFolder);
    
    // open file for writing
    ofstream out(outputFolder + "/" + outputFolder + "_" + tag + ".csv");
    out << "Model";
    bool isfirst = true;

    // iterate through all model folders to extract files
    vector<string> mFolders = getFoldersInFolder(modelFolder);
    for (int i = 0; i < mFolders.size(); i++) {
        string mFolder = mFolders.at(i);

        vector<string> gFolders = getFoldersInFolder(modelFolder + "/" + mFolder + "/");
        for (int j = 0; j < gFolders.size(); j++) {
            string gFolder = gFolders.at(j);

            vector<string> tFolders = getFoldersInFolder(modelFolder + "/" + mFolder + "/" + gFolder + "/");
            cout << "extracting models from " << mFolder << " " << gFolder << endl;
            for (int k = 0; k < tFolders.size(); k++) {
                string tFolder = tFolders.at(k);

                vector<string> fFiles = getFilesInFolder(modelFolder + "/" + mFolder + "/" + gFolder + "/" + tFolder + "/", false);
                for (int l = 0; l < fFiles.size(); l++) {
                    string fFile = fFiles.at(l);

                    // get parts of this file, to check for cloud profile
                    vector<string> parts = split(fFile, '_');
                    string t = parts.at(4).substr(1);
                    string g = parts.at(3).substr(1);
                    string m = parts.at(2).substr(1);
                    string f = parts.at(5).substr(1);

                    // check if cloud profile exists for this model
                    if (!clouds.empty() && !clouds[t + g + m + f])
                        continue;

                    // bin model
                    if (bin) {
                        vector<double> xs; // wavelengths
                        vector<double> ys; // sum of albedos in bin
                        vector<double> ns; // number of data points in bin
                        double start = bins.at(0); // starting wavelength
                        double end = bins.at(1); // ending wavelength
                        double inc = bins.at(2); // bin wavelength size for increments
                        double next = (int)(100000 * (start + inc)) / 100000.0; // keep track of next bin starting value
                        double bin = 0; // current sum for bin
                        int n = 0; // current number of data points in bin

                        // read model
                        ifstream in(modelFolder + "/" + mFolder + "/" + gFolder + "/" + tFolder + "/" + fFile + ".dat");
                        string line = "";
                        getline(in, line);
                        while (getline(in, line)) {
                            // split into <wavelength, albedo value> pair
                            vector<string> cells = split(line, ' ');
                            double wavelength = stod(cells.at(0));
                            double albedo = stod(cells.at(1));

                            // check if we save value
                            if (wavelength < start)
                                continue;
                            if (wavelength >= end)
                                break;
                            if (wavelength == 0.5002 || wavelength == 0.99965)
                                albedo = -1;
                            if (cells.at(1) == "nan")
                                albedo = -1;
                            if (ignoreOutOfBounds && (albedo > 1 || albedo < 0))
                                albedo = -1;
                            // if this is a new value then add to vector and reset next
                            if (wavelength >= next) {
                                xs.push_back(next - inc / 2.0);
                                ys.push_back(bin);
                                ns.push_back(n);
                                next = (int)(100000 * (next + inc)) / 100000.0;
                                bin = 0;
                                n = 0;
                            }

                            // add to bin
                            if (albedo != -1) {
                                bin += albedo;
                                n++;
                            }
                        }

                        // add last bin
                        xs.push_back(next - inc / 2.0);
                        ys.push_back(bin);
                        ns.push_back(n);
                        in.close();

                        // write .csv file
                        if (isfirst) {
                            for (int go = 0; go < xs.size(); go++)
                                out << "," << xs.at(go);
                            out << ",[M/H],log g(cm_s2),T_eff(K),f_sed\n";
                            isfirst = false;
                        }
                        out << fFile;
                        for (int go = 0; go < xs.size(); go++)
                            out << "," << setprecision(7) << ys.at(go) / ns.at(go); // average bin value
                        out << "," << m << "," << g << "," << t << "," << f << "\n";
                    }
                    else { // no binning

                        // read .dat file
                        ifstream in(modelFolder + "/" + mFolder + "/" + gFolder + "/" + tFolder + "/" + fFile + ".dat");
                        string line = "";
                        getline(in, line);
                        vector<double> wavelengths;
                        vector<double> albedos;
                        while (getline(in, line)) {
                            vector<string> cells = split(line, ' ');
                            double wavelength = stod(cells.at(0));
                            double albedo = stod(cells.at(1));

                            if (wavelength == 0.5002 || wavelength == 0.99965)
                                continue;
                            // save as bad value if NaN or Out Of Bounds
                            if (cells.at(1) == "nan")
                                albedo = -1;
                            if ((ignoreOutOfBounds && (albedo > 1 || albedo < 0)))
                                albedo = -1;
                            wavelengths.push_back(wavelength);
                            albedos.push_back(albedo);
                        }
                        in.close();

                        // write .csv file
                        if (isfirst) {
                            for (int go = 0; go < wavelengths.size(); go++)
                                out << "," << wavelengths.at(go);
                            out << ",[M/H],log g(cm_s2),T_eff(K),f_sed\n";
                            isfirst = false;
                        }
                        bool bad = false;
                        for (int i = 0; i < wavelengths.size(); i++) {
                            // linearally interpolate bad values (NaN or O.O.B.)
                            if (albedos.at(i) == -1) {
                                if (i == 0) {
                                    if (albedos.at(i + 1) != -1)
                                        albedos.at(i) = albedos.at(i + 1);
                                    else
                                        bad = true;
                                }
                                else if (i == wavelengths.size() - 1) {
                                    if (albedos.at(i - 1) != -1)
                                        albedos.at(i) = albedos.at(i - 1);
                                    else
                                        bad = true;
                                }
                                else {
                                    if (albedos.at(i + 1) != -1 && albedos.at(i - 1) != -1)
                                        albedos.at(i) = albedos.at(i - 1) + (albedos.at(i + 1) - albedos.at(i - 1)) / (wavelengths.at(i + 1) - wavelengths.at(i - 1)) * (wavelengths.at(i) - wavelengths.at(i - 1));
                                    else
                                        bad = true;
                                }
                            }
                        }
                        //if (!bad) {
                            out << fFile;
                            for (int i = 0; i < albedos.size(); i++)
                                out << "," << setprecision(7) << albedos.at(i); // average bin value
                            out << "," << m << "," << g << "," << t << "," << f << "\n";
                        //}
                        if(bad)
                            cout << "," << m << "," << g << "," << t << "," << f << "\n";
                    }
                }
            }
        }
    }
    out.close();
}

/*
    createDataSets() creates a 1 .csv dataset, split into training, testing, and validation with optionally noisy spectra
    @param1 modelFolder is where models to be read are and where dataset .csv files will be written
    @param2 tag to add on to end of name (e.g. "Degenerate" or "NonDegenerate")
    @param3 trainRatio decimal value [0,1] of ratio of models to be used in training set
    @param4 validationRatio decimal value [0,1] of ratio of models to be used in validation set
    @param5 testRatio decimal value [0,1] of ratio of models to be used in testing set
    @param6 noise is a vector<double> that is either empty to not add noise or contains:
                .at(0) number of noisy models to generate for each model
                .at(1-n) n number of noise levels (multipliers by the average of each model)
*/
void createDataSets(string modelFolder, string tag, double trainRatio, double validationRatio, double testRatio, vector<double> noise) {

    // check parameter validity
    if (modelFolder == "")
        cout << "@param1 in createDataSets() is empty" << endl;
    if (trainRatio <= 0 || trainRatio > 1)
        cout << "@param3 in createDataSets() is invalid, must be (0, 1]" << endl;
    if (validationRatio <= 0 || validationRatio > 1)
        cout << "@param4 in createDataSets() is invalid, must be (0, 1]" << endl;
    if (testRatio <= 0 || testRatio > 1)
        cout << "@param5 in createDataSets() is invalid, must be (0, 1]" << endl;
    if (noise.size() == 1)
        cout << "@param6 in createDataSets() is invalid size, must be either 0 or greater than 1" << endl;
    for (int i = 1; i < noise.size(); i++)
        if (noise.size() == 1)
            if (noise.at(i) <= 0 || noise.at(i) > 1)
                cout << "@param6[ " << i << "] in createDataSets() is invalid, must be (0, 1]" << endl;

    DataSet* baseSet = new DataSet(modelFolder + "/" + modelFolder + "_" + tag + ".csv", NULL);

    // create list of rows to randomly pull from
    vector<int>* poolRows = new vector<int>();
    for (int r = 0; r < baseSet->nRows; r++)
        poolRows->push_back(r);

    // get number of rows for each train, validation, and test set
    int nTrainRows = baseSet->nRows * trainRatio;
    int nValidationRows = baseSet->nRows * validationRatio;
    int nTestRows = baseSet->nRows * testRatio;
    nTrainRows += baseSet->nRows - nTrainRows - nValidationRows - nTestRows;

    // create lambda function to pull rows
    auto pullRows = [](vector<int>* poolRows, int nRows) {
        vector<int> pulledRows;
        for (int i = 0; i < nRows; i++) {
            // get random row from rows vector
            int randIdx = rand() % (int)poolRows->size();
            int randRow = poolRows->at(randIdx);
            poolRows->erase(poolRows->begin() + randIdx);
            pulledRows.push_back(randRow);
        }
        return pulledRows;
    };

    // grab rows for each data set
    vector<int> trainRows = pullRows(poolRows, nTrainRows);
    vector<int> validationRows = pullRows(poolRows, nValidationRows);
    vector<int> testRows = pullRows(poolRows, nTestRows);

    // create lambda function to create datasets
    auto createData = [](string modelFolder, string tag, DataSet* baseSet, vector<double> noise, map<string, DataSet*>* dataSets, string name, vector<int> rows) {
        (*dataSets)[modelFolder + "_" + tag + "_" + name] = new DataSet(baseSet, rows, vector<double>{});
        if (!noise.empty())
            for (int i = 1; i < noise.size(); i++)
                (*dataSets)[modelFolder + "_" + tag + "_" + name + "_Noise" + to_string((int)(noise.at(i) * 100)) + "%"] = new DataSet(baseSet, rows, vector<double>{noise.at(0), noise.at(i)});
    };

    // create data sets
    map<string, DataSet*>* dataSets = new map<string, DataSet*>();
    createData(modelFolder, tag, baseSet, noise, dataSets, "Train", trainRows);
    createData(modelFolder, tag, baseSet, noise, dataSets, "Validation", validationRows);
    createData(modelFolder, tag, baseSet, noise, dataSets, "Test", testRows);
    
    // write data sets
    for (map<string, DataSet*>::iterator it = dataSets->begin(); it != dataSets->end(); it++)
        it->second->write(modelFolder + "/" + it->first);
}

void trainMLP(string outputFolder) {
    string trainingDataFile = "Albedo_Models_248Bins0.3to1.0microns/Albedo_Models_248Bins0.3to1.0microns__DroppedDegeneraciesFsed11_Train_Noise20%.csv";
    string validationDataFile = "Albedo_Models_248Bins0.3to1.0microns/Albedo_Models_248Bins0.3to1.0microns__DroppedDegeneraciesFsed11_Validation_Noise20%.csv";
    string testingDataFile = "Albedo_Models_248Bins0.3to1.0microns/Albedo_Models_248Bins0.3to1.0microns__DroppedDegeneraciesFsed11_Test_Noise20%.csv";
    //string trainingDataFile = "Albedo_Models_Train_500nm_to_1000nm_178Bins.csv";
    //string validationDataFile = "Albedo_Models_Validation_500nm_to_1000nm_178Bins.csv";
    //string testingDataFile = "Albedo_Models_Test_500nm_to_1000nm_178Bins.csv";

    // read from file and create the training, validation, and testing data sets
    DataSet* trainingDataSet = new DataSet(trainingDataFile, NULL);
    DataSet* validationDataSet = new DataSet(validationDataFile, NULL);
    DataSet* testingDataSet = new DataSet(testingDataFile, NULL);

    // check for errors in the data sets
    bool error = false;
    if (trainingDataSet->nCols <= 0) {
        cout << "no datapoints detected in training data, check your .csv file path and make sure the header is formatted properly, see README." << endl;
        error = true;
    }
    else if (trainingDataSet->nLabels <= 0) {
        cout << "no labels detected in training data, check your .csv file path and make sure the header is formatted properly, see README." << endl;
        error = true;
    }
    else {
        if (trainingDataSet->nCols != validationDataSet->nCols) {
            cout << "training data number of columns does not match validation, check your .csv files to insure there are the same number datapoints in each, make sure the header is formatted properly, see README." << endl;
            error = true;
        }
        if (trainingDataSet->nCols != testingDataSet->nCols) {
            cout << "training data number of columns does not match testing, check your .csv files to insure there are the same number datapoints in each, make sure the header is formatted properly, see README." << endl;
            error = true;
        }
        if (trainingDataSet->nLabels != validationDataSet->nLabels) {
            cout << "training number of labels does not match validation, check your .csv files to insure there are the same number of labels in each, make sure the header is formatted properly, see README." << endl;
            error = true;
        }
        if (trainingDataSet->nLabels != testingDataSet->nLabels) {
            cout << "training number of labels does not match testing, check your .csv files to insure there are the same number of labels in each, make sure the header is formatted properly, see README." << endl;
            error = true;
        }
    }
    if (error) {
        cout << "Program has stopped, press any key and return to quit." << endl;
        return;
    }

    // create sub folder path for all MLP training results - by epoch
    //string subFolder = outputFolder + "/Results_For_Each_Epoch/";
    //createFolder(subFolder);

    // set mlp parameters
    int nEpochs = 1000; // max number of epochs to train MLP for
    vector<int> layers; // layers specify the number of nodes in each layer
    layers.push_back(trainingDataSet->nCols); // number of input nodes
    layers.push_back(64); // number of hidden layer 1 nodes
    layers.push_back(64); // number of hidden layer 2 nodes
    layers.push_back(32); // number of hidden layer 3 nodes
    layers.push_back(trainingDataSet->nLabels); // number of output nodes
    vector<ActFunc*> actFuncs; // activation functions for hidden and ouput nodes
    actFuncs.push_back(ELU::get()); // hidden layer 1
    actFuncs.push_back(ELU::get()); // hidden layer 2
    actFuncs.push_back(ELU::get()); // hidden layer 3
    actFuncs.push_back(SIG::get()); // output layer
    int batchSize = 128;
    double learningMomentum = 0.9;
    double learningRate = 1.0;
    bool cyclicLearning = true;
    double rateMin = 0.1;
    double rateMax = 0.9;
    double rateStepSize = 2.0 * (double)trainingDataSet->nRows / (double)batchSize; // "ideal" adaptive step size
    bool dropout = true;
    double inputProb = 0.9;
    double hiddenProb = 1.0;
    double L1 = 1E-7;
    double L2 = 1E-6;
    double maxNorm = 1.5;

    // toggle these booleans on or off to alter what gets output from trianing
    bool checkTrainSet = true; // predict the train set after each epoch
    bool outputTrainPredictions = false; // output the raw data and exact MLP predictions (output) for each training vector to a .csv file
    bool outputValPredictions = true; // output the raw data and exact MLP predictions (output) for each validation vector to a .csv file
    bool outputMLPParams = true; // output the trained MLP parameters and set hyper-parameters, you can also read this in whenever to get an                                old mlp back in action, continue training at another time, etc...
    bool outputResults = true; // writes the error metric (accuracy, RMSE, etc) results for each epoch to a .csv file
    bool classBinary = false; // classify each vector as one class each, based on predictions (output values) from MLP, and calculate error
    bool regress = true; // calculate error, assuming regression. This is the RMSE measured from the difference between predictions and labels
    bool classMulti = false; // classify each vector as multiple classes, based on if the prediction is higher than multiProb, and calc err.
    double multiProb = 0.5; // only if using multi-classification, this is the value that a prediction must be greater than or equal to so that it belongs to a class

    // create the MLP object from scratch with specified layers and random weights
    MLP* mlp = new MLP(layers);

    // set activation functions for each layer
    for (int i = 0; i < actFuncs.size(); i++)
        mlp->setActFuncs(i + 1, actFuncs.at(i));

    // normalize labels between 0.01 and 0.99
    mlp->normalize(trainingDataSet);
    mlp->normalize(validationDataSet);

    // set other MLP hyper-paremeters
    mlp->batchSize = batchSize;
    mlp->learningRate = learningRate;
    mlp->learningMomentum = learningMomentum;
    mlp->cyclicLearning = cyclicLearning;
    mlp->rateMin = rateMin;
    mlp->rateMax = rateMax;
    mlp->rateStepSize = rateStepSize;
    mlp->setDropOut(dropout, inputProb, hiddenProb);
    mlp->L1 = L1;
    mlp->L2 = L2;
    if (maxNorm > 0.0)
        mlp->maxNorm = true;
    mlp->radiusLength = maxNorm;

    cout << "MLP created! Results are actively written to a .csv file next to where your train data file is after each epoch. The MLP that yielded the lowest validation RMSE will be actively overwritten next to where your training data file is along with testing predictions from that MLP. Training... On Epoch #:" << endl;

    // keep track of error metrics for the different sets. These output the results for each epoch
    map<int, map<int, map<string, double> > > binTrainErr;
    map<int, map<int, map<string, double> > > multiTrainErr;
    map<int, map<int, double> > regressionTrainRMSE;
    map<int, map<int, map<string, double> > > binValErr;
    map<int, map<int, map<string, double> > > multiValErr;
    map<int, map<int, double> > regressionValRMSE;

    // keep track of best epoch
    double bestRMSE = 1.0;
    int bestEpoch = 0;

    // create a stopwatch to time training
    StopWatch stopwatch;
    for (int epoch = 1; epoch <= nEpochs; epoch++) {
        cout << epoch << " ";

        // create a folder to output results for this epoch
        //string epochFolder = subFolder + "Epoch" + to_string(epoch) + "/";
        //createFolder(epochFolder);

        // train MLP
        mlp->train(trainingDataSet);

        // get predictions and calc error for this epoch
        if (checkTrainSet) {
            mlp->predict(trainingDataSet, false);
            if (classBinary)
                binTrainErr[epoch] = binaryClassify(trainingDataSet);
            if (classMulti)
                multiTrainErr[epoch] = multiClassify(trainingDataSet, multiProb);
            if (regress)
                regressionTrainRMSE[epoch] = regression(trainingDataSet);
            //if (outputTrainPredictions)
                //trainingDataSet->write(epochFolder + "Train_Set_Predictions.csv");
        }

        mlp->predict(validationDataSet, false);
        if (classBinary) {
            binValErr[epoch] = binaryClassify(validationDataSet);
            binaryClassify(testingDataSet);
        }
        if (classMulti) {
            multiValErr[epoch] = multiClassify(validationDataSet, multiProb);
            multiClassify(testingDataSet, multiProb);
        }

        //if (outputValPredictions)
            //validationDataSet->write(epochFolder + "Validation_Predictions");
        //testingDataSet->write(epochFolder + "Test_Predictions");

        regressionValRMSE[epoch] = regression(validationDataSet);
        if (regressionValRMSE[epoch][-1] < bestRMSE) {
            bestRMSE = regressionValRMSE[epoch][-1];
            remove((outputFolder + "/MLP_Epoch" + to_string(bestEpoch) + ".csv").c_str());
            remove((outputFolder + "/Test_Predictions.csv").c_str());
            bestEpoch = epoch;
            mlp->write(outputFolder + "/MLP_Epoch" + to_string(epoch));
            mlp->predict(testingDataSet, true);
            testingDataSet->write(outputFolder + "/Test_Predictions");
        }

        // output MLP parameters to file
        //if (outputMLPParams)
            //mlp->write(epochFolder + "MLP");

        // output results
        if (outputResults) {
            if (classBinary) {
                if (checkTrainSet)
                    writeClassificationResults(binTrainErr, trainingDataSet->labelNames, trainingDataSet->nLabels, outputFolder + "/BinaryClass_Train_Results");
                writeClassificationResults(binValErr, trainingDataSet->labelNames, trainingDataSet->nLabels, outputFolder + "/BinaryClass_Validation_Results");
            }
            if (classMulti) {
                if (checkTrainSet)
                    writeClassificationResults(multiTrainErr, trainingDataSet->labelNames, trainingDataSet->nLabels, outputFolder + "/MultiClass_Train_Results");
                writeClassificationResults(multiValErr, trainingDataSet->labelNames, trainingDataSet->nLabels, outputFolder + "/MultiClass_Validation_Results");
            }
            if (regress) {
                if (checkTrainSet)
                    writeRegressionResults(regressionTrainRMSE, trainingDataSet->labelNames, trainingDataSet->nLabels, outputFolder + "/Regression_Train_Results");
                writeRegressionResults(regressionValRMSE, trainingDataSet->labelNames, trainingDataSet->nLabels, outputFolder + "/Regression_Validation_Results");
            }
        }

        // output progress every 100 epochs
        if (epoch % 100 == 0)
            cout << endl << "Finished the previous 100 epochs in " << stopwatch.lap() << " ms" << endl;
    }

    // all done!
    cout << endl << "Mission complete in " << stopwatch.stop() << " ms. Check the folder where your training data file was to see results!" << endl;

}

/*
    checks spectra in folder for degeneracies
    writes an output .csv file tallying the number of degenerate and non-degenerate spectra
    writes an output .csv DataSet file with all nondegenerate models
*/
void checkForDegeneracies(string modelFolder, string tag) {
    // open dataset
    DataSet* dataset = new DataSet(modelFolder + "/" + modelFolder + "_" + tag + ".csv", NULL);

    // find degeneracies
    vector< pair<int, int> > degeneracies;
    for (int r = 0; r < dataset->nRows; r++) {
        for (int r2 = r - 1; r2 >= 0; r2--) {
            bool match = true;
            for (int c = 0; c < dataset->nCols && match; c++) {
                if (dataset->data[r * dataset->nCols + c] != dataset->data[r2 * dataset->nCols + c])
                    match = false;
            }
            if (match)
                degeneracies.push_back(pair<int, int>{r, r2});
        }
    }

    // merge degeneracies
    vector< vector<int> > mergedDegeneracies;
    for (int i = 0; i < degeneracies.size(); i++) {
        int r1i = degeneracies.at(i).first;
        int r2i = degeneracies.at(i).second;
        bool merged = false;
        for (int j = 0; j < mergedDegeneracies.size() && !merged; j++) {
            for (int k = 0; k < mergedDegeneracies.at(j).size() && !merged; k++) {
                int r = mergedDegeneracies.at(j).at(k);
                if (r == r1i || r == r2i) {
                    bool added = false;
                    if (r == r1i) {
                        for (int l = 0; l < mergedDegeneracies.at(j).size() && !added; l++)
                            if (r2i == mergedDegeneracies.at(j).at(l))
                                added = true;
                        if (!added)
                            mergedDegeneracies.at(j).push_back(r2i);
                    }
                    if (r == r2i) {
                        for (int l = 0; l < mergedDegeneracies.at(j).size() && !added; l++)
                            if (r1i == mergedDegeneracies.at(j).at(l))
                                added = true;
                        if (!added)
                            mergedDegeneracies.at(j).push_back(r1i);
                    }
                    merged = true;
                }
            }
        }
        if (!merged)
            mergedDegeneracies.push_back(vector<int>{r1i, r2i});
    }

    // remove degeneracies
    vector<int> rows;
    for (int r = 0; r < dataset->nRows; r++)
        rows.push_back(r);
    int nSameMGT = 0;
    int nDiffMGT = 0;
    for (int i = 0; i < mergedDegeneracies.size(); i++) {
        vector<double> ms;
        vector<double> gs;
        vector<double> ts;
        vector<double> fs;
        bool sameMGT = true;
        for (int j = 0; j < mergedDegeneracies.at(i).size(); j++) {
            int mR = mergedDegeneracies.at(i).at(j);
            double m = dataset->labels[mR * dataset->nLabels + 0];
            ms.push_back(m);
            double g = dataset->labels[mR * dataset->nLabels + 1];
            gs.push_back(g);
            double t = dataset->labels[mR * dataset->nLabels + 2];
            ts.push_back(t);
            double f = dataset->labels[mR * dataset->nLabels + 3];
            fs.push_back(f);
            for (int k = j - 1; k >= 0 && sameMGT; k--) {
                int mR2 = mergedDegeneracies.at(i).at(k);
                if (m != dataset->labels[mR2 * dataset->nLabels + 0]
                    || g != dataset->labels[mR2 * dataset->nLabels + 1]
                    || t != dataset->labels[mR2 * dataset->nLabels + 2]) {
                    cout << m << " " << g << " " << t << " "
                        << dataset->labels[mR2 * dataset->nLabels + 0] << " "
                        << dataset->labels[mR2 * dataset->nLabels + 1] << " "
                        << dataset->labels[mR2 * dataset->nLabels + 2] << " " << endl;
                    nDiffMGT++;
                    cout << dataset->samples[mR] << endl;
                    sameMGT = false;
                }
            }

            for (int r = 0; r < rows.size(); r++) {
                if (rows.at(r) == mR) {
                    rows.erase(rows.begin() + r);
                    break;
                }
            }
        }
        if (sameMGT)
            nSameMGT++;
    }
    DataSet* noDegeneracies = new DataSet(dataset, rows, vector<double>{});
    noDegeneracies->write(modelFolder + "/" + modelFolder + "_" + tag + "_NonDegenerate");

    // output tallies
    ofstream out(modelFolder + "/" + modelFolder + "_" + tag + "_Degeneracy_List.csv");
    out << "There are " << dataset->nRows - rows.size() << " degenerate models.\n";
    out << "There are " << nSameMGT << " sets of degenerate models with same M g T values but different f values.\n";
    out << "There are " << nDiffMGT << " sets of degenerate models with other varying parameters.\n\n";
    out << "Number of Degeneracies,Models...\n";
    for (int i = 0; i < mergedDegeneracies.size(); i++) {
        out << mergedDegeneracies.at(i).size();
        for (int j = 0; j < mergedDegeneracies.at(i).size(); j++)
            out << "," << dataset->samples[mergedDegeneracies.at(i).at(j)];
        out << "\n";
    }
    out.close();
}

/*
    drops degenerate models with highest f_sed
    writes an output .csv DataSet file with surviving models
    setFsed11 to true to keep one model for each set of degenerate models and set its fsed value to 11
*/
void dropDegeneraciesWithHighestFsed(string modelFolder, string tag, bool setFsed11) {
    // open dataset
    DataSet* dataset = new DataSet(modelFolder + "/" + modelFolder + "_" + tag + ".csv", NULL);
    
    // open degnerate model list and models to drop
    vector< vector<double> > setModels;
    vector< vector<double> > dropModels;
    ifstream in(modelFolder + "/" + modelFolder + "_" + tag + "_Degeneracy_List.csv");
    string line = "";
    getline(in, line);
    getline(in, line);
    getline(in, line);
    getline(in, line);
    getline(in, line);
    while (getline(in, line)) {
        vector< vector<double> > params;
        double minF = 10;
        vector<string> cells = split(line, ',');
        for (int i = 1; i < cells.size(); i++) {
            vector<string> parts = split(cells.at(i), '_');
            double m = stod(parts.at(2).substr(1));
            double g = stod(parts.at(3).substr(1));
            double t = stod(parts.at(4).substr(1));
            double f = stod(parts.at(5).substr(1));
            minF = min(f, minF);
            params.push_back(vector<double>{m, g, t, f});
        }
        for (int i = 0; i < params.size(); i++) {
            if (params.at(i).at(3) == minF) {
                setModels.push_back(params.at(i));
                continue;
            }
            dropModels.push_back(params.at(i));
        }
    }
    in.close();

    // get rows to keep
    vector<int> rows;
    for (int r = 0; r < dataset->nRows; r++) {
        double m = dataset->labels[r * dataset->nLabels + 0];
        double g = dataset->labels[r * dataset->nLabels + 1];
        double t = dataset->labels[r * dataset->nLabels + 2];
        double f = dataset->labels[r * dataset->nLabels + 3];

        bool drop = false;
        for (int i = 0; i < dropModels.size() && !drop; i++)
            if (dropModels.at(i).at(0) == m
                && dropModels.at(i).at(1) == g
                && dropModels.at(i).at(2) == t
                && dropModels.at(i).at(3) == f)
                drop = true;
        if (!drop)
            rows.push_back(r);

        if (setFsed11) {
            bool set = false;
            for (int i = 0; i < setModels.size() && !set; i++)
                if (setModels.at(i).at(0) == m
                    && setModels.at(i).at(1) == g
                    && setModels.at(i).at(2) == t
                    && setModels.at(i).at(3) == f)
                    set = true;
            if (set)
                dataset->labels[r * dataset->nLabels + 3] = 11;
        }
    }

    DataSet* dropped = new DataSet(dataset, rows, vector<double>{});
    dropped->write(modelFolder + "/" + modelFolder + "_" + tag + "_DroppedDegeneracies" + (setFsed11 ? "Fsed11" : ""));
}

/*
    drops degenerate models with highest f_sed
    writes an output .csv DataSet file with surviving models
*/
void addBinaryCloudClass(string modelFolder, string tag) {
    // open dataset
    DataSet* dataset = new DataSet(modelFolder + "/" + modelFolder + "_" + tag + ".csv", NULL);

    // open degnerate model list and models to drop
    vector< vector<double> > dropModels;
    vector< vector<double> > noCloudModels;
    ifstream in(modelFolder + "/" + modelFolder + "_" + tag + "_Degeneracy_List.csv");
    string line = "";
    getline(in, line);
    getline(in, line);
    getline(in, line);
    getline(in, line);
    getline(in, line);
    while (getline(in, line)) {
        vector< vector<double> > params;
        double minF = 10;
        vector<string> cells = split(line, ',');
        for (int i = 1; i < cells.size(); i++) {
            vector<string> parts = split(cells.at(i), '_');
            double m = stod(parts.at(2).substr(1));
            double g = stod(parts.at(3).substr(1));
            double t = stod(parts.at(4).substr(1));
            double f = stod(parts.at(5).substr(1));
            minF = min(f, minF);
            params.push_back(vector<double>{m, g, t, f});
        }
        for (int i = 0; i < params.size(); i++) {
            if (params.at(i).at(3) == minF) {
                noCloudModels.push_back(params.at(i));
                continue;
            }
            dropModels.push_back(params.at(i));
        }
    }
    in.close();

    // get rows to keep
    vector<int> rows;
    for (int r = 0; r < dataset->nRows; r++) {
        double m = dataset->labels[r * dataset->nLabels + 0];
        double g = dataset->labels[r * dataset->nLabels + 1];
        double t = dataset->labels[r * dataset->nLabels + 2];
        double f = dataset->labels[r * dataset->nLabels + 3];

        bool drop = false;
        for (int i = 0; i < dropModels.size() && !drop; i++)
            if (dropModels.at(i).at(0) == m
                && dropModels.at(i).at(1) == g
                && dropModels.at(i).at(2) == t
                && dropModels.at(i).at(3) == f)
                drop = true;
        if (!drop)
            rows.push_back(r);
    }

    // set cloud class
    double* labels = new double[dataset->nRows * (dataset->nLabels + 1)];
    for (int r = 0; r < dataset->nRows; r++) {
        double m = dataset->labels[r * dataset->nLabels + 0];
        double g = dataset->labels[r * dataset->nLabels + 1];
        double t = dataset->labels[r * dataset->nLabels + 2];
        double f = dataset->labels[r * dataset->nLabels + 3];

        for (int l = 0; l < dataset->nLabels; l++)
            labels[dataset->nRows * (dataset->nLabels + 1) + l] = dataset->labels[dataset->nRows * dataset->nLabels + l];

        bool drop = false;
        for (int i = 0; i < dropModels.size() && !drop; i++)
            if (dropModels.at(i).at(0) == m
                && dropModels.at(i).at(1) == g
                && dropModels.at(i).at(2) == t
                && dropModels.at(i).at(3) == f)
                drop = true;
        if (!drop)
            rows.push_back(r);
    }

    DataSet* dropped = new DataSet(dataset, rows, vector<double>{});
    dropped->write(modelFolder + "/" + modelFolder + "_" + tag + "_DroppedDegeneracies");
}

void trainMLP_UI() {
    // prompt user to input training data file path
    string trainingDataFile = "";
    cout << "Type full path to .csv training data file (including file extension) and press return:" << endl;
    trainingDataFile = readString();

    // prompt user to input validation data file path
    string validationDataFile = "";
    cout << "Type full path to .csv validation data file (including file extension) and press return:" << endl;
    validationDataFile = readString();

    // prompt user to input testing data file path
    string testingDataFile = "";
    cout << "Type full path to .csv testing data file (including file extension) and press return:" << endl;
    testingDataFile = readString();

    // read from file and create the training, validation, and testing data sets
    DataSet* trainingDataSet = new DataSet(trainingDataFile, NULL);
    DataSet* validationDataSet = new DataSet(validationDataFile, NULL);
    DataSet* testingDataSet = new DataSet(testingDataFile, NULL);

    // check for errors in the data sets
    bool error = false;
    if (trainingDataSet->nCols <= 0) {
        cout << "no datapoints detected in training data, check your .csv file path and make sure the header is formatted properly, see README." << endl;
        error = true;
    }
    else if (trainingDataSet->nLabels <= 0) {
        cout << "no labels detected in training data, check your .csv file path and make sure the header is formatted properly, see README." << endl;
        error = true;
    }
    else {
        if (trainingDataSet->nCols != validationDataSet->nCols) {
            cout << "training data number of columns does not match validation, check your .csv files to insure there are the same number datapoints in each, make sure the header is formatted properly, see README." << endl;
            error = true;
        }
        if (trainingDataSet->nCols != testingDataSet->nCols) {
            cout << "training data number of columns does not match testing, check your .csv files to insure there are the same number datapoints in each, make sure the header is formatted properly, see README." << endl;
            error = true;
        }
        if (trainingDataSet->nLabels != validationDataSet->nLabels) {
            cout << "training number of labels does not match validation, check your .csv files to insure there are the same number of labels in each, make sure the header is formatted properly, see README." << endl;
            error = true;
        }
        if (trainingDataSet->nLabels != testingDataSet->nLabels) {
            cout << "training number of labels does not match testing, check your .csv files to insure there are the same number of labels in each, make sure the header is formatted properly, see README." << endl;
            error = true;
        }
    }
    if (error) {
        cout << "Program has stopped, press any key and return to quit." << endl;
        return;
    }

    // get path to folder that training data file is in
    string folder = trainingDataFile;
    bool truncated = false;
    for (int c = trainingDataFile.length() - 1; c >= 0; c--)
        if (trainingDataFile.at(c) == '/' || trainingDataFile.at(c) == '\\') {
            folder = trainingDataFile.substr(0, c) + "/";
            truncated = true;
            break;
        }
    if (!truncated)
        folder = "";

    // create sub folder path for all MLP training results - by epoch
    string subFolder = folder + "Training_Results/";
    createFolder(subFolder);

    // prompt if user wants to use default MLP or create their own
    string response = "";
    while (response == "") {
        cout << "Would you like to create your own MLP or use the default one that will auto detect the number of inputs and ouputs from the training data and use a default MLP as detailed in the paper (no noise)? Enter 'own' or 'auto' respectuflly." << endl;
        cin >> response;
        if (response != "own" && response != "auto") {
            cout << "invalid response..." << endl;
            response = "";
        }
    }

    // default mlp parameters
    int nEpochs = 1000; // max number of epochs to train MLP for
    vector<int> layers; // layers specify the number of nodes in each layer
    layers.push_back(trainingDataSet->nCols); // number of input nodes
    layers.push_back(64); // number of hidden layer 1 nodes
    layers.push_back(64); // number of hidden layer 2 nodes
    layers.push_back(32); // number of hidden layer 3 nodes
    layers.push_back(trainingDataSet->nLabels); // number of output nodes
    vector<ActFunc*> actFuncs; // activation functions for hidden and ouput nodes
    actFuncs.push_back(ELU::get()); // hidden layer 1
    actFuncs.push_back(ELU::get()); // hidden layer 2
    actFuncs.push_back(ELU::get()); // hidden layer 3
    actFuncs.push_back(SIG::get()); // output layer
    int batchSize = 128;
    double learningMomentum = 0.9;
    double learningRate = 1.0;
    bool cyclicLearning = true;
    double rateMin = 0.1;
    double rateMax = 0.9;
    double rateStepSize = 2.0 * (double)trainingDataSet->nRows / (double)batchSize; // "ideal" adaptive step size
    bool dropout = false;
    double inputProb = 1.0;
    double hiddenProb = 1.0;
    double L1 = 1E-7;
    double L2 = 1E-5;
    double maxNorm = 6.1;

    // toggle these booleans on or off to alter what gets output from trianing
    bool checkTrainSet = true; // predict the train set after each epoch and track results
    bool outputTrainPredictions = false; // output the raw data and exact MLP predictions (output) for each training vector to a .csv file
    bool outputValPredictions = false; // output the raw data and exact MLP predictions (output) for each validation vector to a .csv file
    bool outputTestPredictions = false; // output the raw data and exact MLP predictions (output) for each test vector to a .csv file
    bool outputMLPParams = false; // output the trained MLP parameters and set hyper-parameters, you can also read this in whenever to get an                                old mlp back in action, continue training at another time, etc...
    bool outputResults = true; // writes the error metric (accuracy, RMSE, etc) results for each epoch to a .csv file
    bool makeEpochFolder = (outputTrainPredictions || outputValPredictions || outputTestPredictions || outputMLPParams);
    
    // these booleans are used for the type of training (binary class, regress, multi class)
    bool classBinary = false; // classify each vector as one class each, based on predictions (output values) from MLP, and calculate error
    bool regress = true; // calculate error, assuming regression. This is the RMSE measured from the difference between predictions and labels
    bool classMulti = false; // classify each vector as multiple classes, based on if the prediction is higher than multiProb, and calc err.
    double multiProb = 0.5; // only if using multi-classification, this is the value that a prediction must be greater than or equal to so that it belongs to a class

    // check if user opted to make their own MLP
    if (response == "own") {
        cout << "how many training epochs?" << endl;
        cin >> nEpochs;

        int nH = 0;
        cout << "how many hidden layers?" << endl;
        cin >> nH;
        layers.resize(2 + nH);
        actFuncs.resize(1 + nH);
        int n = 0;
        for (int i = 0; i < nH; i++) {
            cout << "how many nodes in hidden layer " << (i + 1) << "?" << endl;
            cin >> n;
            layers.at(i + 1) = n;

            response = "";
            while (response == "") {
                cout << "what type of actication function for hidden layer " << (i + 1) << "? 'elu' for Exponential Linear Units (suggested for hidden layers) or 'sig' for Sigmoid..." << endl;
                cin >> response;
                if (response != "elu" && response != "sig") {
                    cout << "invalid response..." << endl;
                    response = "";
                }
                else {
                    if (response == "elu")
                        actFuncs.at(i) = ELU::get();
                    else if (response == "sig")
                        actFuncs.at(i) = SIG::get();
                }
            }
        }

        response = "";
        while (response == "") {
            cout << "what type of actication function for the output layer? 'elu' for Exponential Linear Units or 'sig' for Sigmoid (suggested for output layer)..." << endl;
            cin >> response;
            if (response != "elu" && response != "sig") {
                cout << "invalid response..." << endl;
                response = "";
            }
            else {
                if (response == "elu")
                    actFuncs.at(nH) = ELU::get();
                else if (response == "sig")
                    actFuncs.at(nH) = SIG::get();
            }
        }

        cout << "how many training vectors in each mini-batch to parellelize training? Enter '1' to not parallelize training." << endl;
        cin >> batchSize;

        cout << "Enter a learning momentum floating point value greater than or equal to 0.0 (Enter '0' to not use learning momentum):" << endl;
        cin >> learningMomentum;

        cout << "Enter a learning rate floating point value greater than 0.0 (Enter '1' to not use learning rate):" << endl;
        cin >> learningRate;

        response = "";
        while (response == "") {
            cout << "Use cyclic learning? Enter 'yes' or 'no'." << endl;
            cin >> response;
            if (response != "yes" && response != "no") {
                cout << "invalid response..." << endl;
                response = "";
            }
            else {
                if (response == "yes") {
                    cyclicLearning = true;
                    cout << "Enter the floating point minimum learning rate for cyclic learning, greater than 0.0:" << endl;
                    cin >> rateMin;
                    cout << "Enter the floating point maximum learning rate for cyclic learning, greater than 0.0:" << endl;
                    cin >> rateMax;
                    cout << "Enter the integer number of epochs for step-size that determines the cycle rate in cycle learning, greater than or equal to 1:" << endl;
                    cin >> rateStepSize;
                    rateStepSize *= (double)trainingDataSet->nRows / (double)batchSize;

                }
                else if (response == "no")
                    cyclicLearning = false;
            }
        }

        response = "";
        while (response == "") {
            cout << "Use dropout? Enter 'yes' or 'no'." << endl;
            cin >> response;
            if (response != "yes" && response != "no") {
                cout << "invalid response..." << endl;
                response = "";
            }
            else {
                if (response == "yes") {
                    dropout = true;
                    cout << "Enter the floating point probability of turning an input node on, (0.0, 1.0] :" << endl;
                    cin >> inputProb;
                    cout << "Enter the floating point probability of turning a hidden node on, (0.0, 1.0] :" << endl;
                    cin >> hiddenProb;

                }
                else if (response == "no")
                    dropout = false;
            }
        }

        cout << "Enter the coefficient for L1 regularization, a floating point value >= 0.0 (Enter '0.0' to not use L1):" << endl;
        cin >> L1;

        cout << "Enter the coefficient for L2 regularization, a floating point value >= 0.0 (Enter '0.0' to not use L2):" << endl;
        cin >> L2;

        cout << "Enter the radius length for max-norm regularization, a floating point value >= 0.0 (Enter '0.0' to not use max-norm):" << endl;
        cin >> maxNorm;

        response = "";
        while (response == "") {
            cout << "Do you want to output results for using the MLP as a regressor (outputting continous values for each output parameter)? Enter 'yes' or 'no'." << endl;
            cin >> response;
            if (response != "yes" && response != "no") {
                cout << "invalid response..." << endl;
                response = "";
            }
            else {
                if (response == "yes")
                    regress = true;
                else if (response == "no")
                    regress = false;
            }
        }

        response = "";
        while (response == "") {
            cout << "Do you want to output results for using the MLP as a binary classifier (classifying each vector as the output class with highest confidence score)? Enter 'yes' or 'no'." << endl;
            cin >> response;
            if (response != "yes" && response != "no") {
                cout << "invalid response..." << endl;
                response = "";
            }
            else {
                if (response == "yes")
                    classBinary = true;
                else if (response == "no")
                    classBinary = false;
            }
        }

        response = "";
        while (response == "") {
            cout << "Do you want to output results for using the MLP as a multi classifier (classifying each vector as multiple classes by being positive for a class if the confidence score is higher than a given threshold which you will enter if you enter 'yes')? Enter 'yes' or 'no'." << endl;
            cin >> response;
            if (response != "yes" && response != "no") {
                cout << "invalid response..." << endl;
                response = "";
            }
            else {
                if (response == "yes") {
                    classMulti = true;
                    cout << "Enter the threshold confidence score for an input vector to be positive for a class, a floating point value bounded by the activation function's range you chose to use in the output layer; for sigmoid this is (0,1):" << endl;
                    cin >> multiProb;

                }
                else if (response == "no")
                    classMulti = false;
            }
        }

        response = "";
        while (response == "") {
            cout << "Would you like to multithread the training process, to speed it up? This is only possible if your hardware can handle it and you selected a mini-batch size greater than 1. Enter 'yes' or 'no'." << endl;
            cin >> response;
            if (response != "yes" && response != "no") {
                cout << "invalid response..." << endl;
                response = "";
            }
            else {
                if (response == "yes") {
                    canMultiThread = true;
                    cout << "Enter a positive integer value specifying the number of threads you want to use, this is limited by the hardware you are using:" << endl;
                    cin >> nGlobalThreads;

                }
                else if (response == "no")
                    canMultiThread = false;
            }
        }
    }

    // create the MLP object from scratch with specified layers and random weights
    MLP* mlp = new MLP(layers);

    // set activation functions for each layer
    for (int i = 0; i < actFuncs.size(); i++)
        mlp->setActFuncs(i + 1, actFuncs.at(i));

    // normalize labels between 0.01 and 0.99
    mlp->normalize(trainingDataSet);
    mlp->normalize(validationDataSet);
    mlp->normalize(testingDataSet);

    // set other MLP hyper-paremeters
    mlp->batchSize = batchSize;
    mlp->learningRate = learningRate;
    mlp->learningMomentum = learningMomentum;
    mlp->cyclicLearning = cyclicLearning;
    mlp->rateMin = rateMin;
    mlp->rateMax = rateMax;
    mlp->rateStepSize = rateStepSize;
    mlp->setDropOut(dropout, inputProb, hiddenProb);
    mlp->L1 = L1;
    mlp->L2 = L2;
    if (maxNorm > 0.0)
        mlp->maxNorm = true;
    mlp->radiusLength = maxNorm;

    cout << "MLP created! Results are actively written to a .csv file next to where your train data file is after each epoch. The MLP that yielded the lowest validation RMSE will be actively overwritten next to where your training data file is along with testing predictions from that MLP. Training... On Epoch #:" << endl;

    // keep track of error metrics for the different sets. These output the results for each epoch
    map<int, map<int, map<string, double> > > binTrainErr;
    map<int, map<int, map<string, double> > > multiTrainErr;
    map<int, map<int, double> > regressionTrainRMSE;
    map<int, map<int, map<string, double> > > binValErr;
    map<int, map<int, map<string, double> > > multiValErr;
    map<int, map<int, double> > regressionValRMSE;

    // keep track of best epoch
    double bestRMSE = 1.0;
    int bestEpoch = 0;

    // create a stopwatch to time training
    StopWatch stopwatch;
    for (int epoch = 1; epoch <= nEpochs; epoch++) {
        cout << epoch << " ";

        // create a folder to output results for this epoch
        string epochFolder = subFolder + "Epoch" + to_string(epoch) + "/";
        if (makeEpochFolder)
            createFolder(epochFolder);

        // train MLP
        mlp->train(trainingDataSet);

        // get predictions and calc error for this epoch
        if (checkTrainSet) {
            mlp->predict(trainingDataSet, false);
            if (classBinary)
                binTrainErr[epoch] = binaryClassify(trainingDataSet);
            if (classMulti)
                multiTrainErr[epoch] = multiClassify(trainingDataSet, multiProb);
            if (regress)
                regressionTrainRMSE[epoch] = regression(trainingDataSet);
            if (outputTrainPredictions)
                trainingDataSet->write(epochFolder + "Train_Set_Predictions.csv");
        }

        mlp->predict(validationDataSet, false);
        mlp->predict(testingDataSet, false);
        if (classBinary) {
            binValErr[epoch] = binaryClassify(validationDataSet);
            binaryClassify(testingDataSet);
        }
        if (classMulti) {
            multiValErr[epoch] = multiClassify(validationDataSet, multiProb);
            multiClassify(testingDataSet, multiProb);
        }
        mlp->predict(testingDataSet, true);

        if (outputValPredictions)
            validationDataSet->write(epochFolder + "Validation_Predictions");
        if (outputTestPredictions)
            testingDataSet->write(epochFolder + "Test_Predictions");

        regressionValRMSE[epoch] = regression(validationDataSet);
        if (regressionValRMSE[epoch][-1] < bestRMSE) {
            bestRMSE = regressionValRMSE[epoch][-1];
            remove((subFolder + "MLP_Epoch" + to_string(bestEpoch) + ".csv").c_str());
            remove((subFolder + "Test_Predictions.csv").c_str());
            bestEpoch = epoch;
            mlp->write(subFolder + "MLP_Epoch" + to_string(epoch));
            testingDataSet->write(subFolder + "Test_Predictions");
        }

        // output MLP parameters to file
        if (outputMLPParams)
            mlp->write(epochFolder + "MLP");

        // output results
        if (outputResults) {
            if (classBinary) {
                if (checkTrainSet)
                    writeClassificationResults(binTrainErr, trainingDataSet->labelNames, trainingDataSet->nLabels, subFolder + "BinaryClass_Train_Results");
                writeClassificationResults(binValErr, trainingDataSet->labelNames, trainingDataSet->nLabels, subFolder + "BinaryClass_Validation_Results");
            }
            if (classMulti) {
                if (checkTrainSet)
                    writeClassificationResults(multiTrainErr, trainingDataSet->labelNames, trainingDataSet->nLabels, subFolder + "MultiClass_Train_Results");
                writeClassificationResults(multiValErr, trainingDataSet->labelNames, trainingDataSet->nLabels, subFolder + "MultiClass_Validation_Results");
            }
            if (regress) {
                if (checkTrainSet)
                    writeRegressionResults(regressionTrainRMSE, trainingDataSet->labelNames, trainingDataSet->nLabels, subFolder + "Regression_Train_Results");
                writeRegressionResults(regressionValRMSE, trainingDataSet->labelNames, trainingDataSet->nLabels, subFolder + "Regression_Validation_Results");
            }
        }

        // output progress every 100 epochs
        if (epoch % 100 == 0)
            cout << endl << "Finished the previous 100 epochs in " << stopwatch.lap() << " ms" << endl;
    }

    // all done!
    cout << endl << "Mission complete in " << stopwatch.stop() << " ms. Check the folder where your training data file was to see results!" << endl;

}
void testMLP_UI() {
    // path to data file (do not put file extension in name) reads from .csv
    string testDataPath = "/Users/tjohnsen/Documents/GitHub/MLP-Estimating-Exoplanet-Parameters/Binned_Model_Datasets/Albedo_Models_248Bins0.3to1.0microns__DroppedDegeneraciesFsed11_Test_Noise5%.csv"; // test data .csv file
    cout << "Enter full file path to Spectra (with file extension):" << endl;
    //cin >> testDataPath;
    
    bool addNoise = false;
    double noise = 0.0;
    if(testDataPath == "noise") {
        addNoise = true;
        cout << "What noise level? Example: '20' for 20% gaussian noise:" << endl;
        cin >> noise;
        cout << "Enter full file path to Spectra (with file extension):" << endl;
        cin >> testDataPath;
    }
    
    // create data object
    DataSet* testDataSet = new DataSet(testDataPath, NULL);
    if(addNoise)
        testDataSet->addNoise(noise);
    
    // path to pretrained MLP (do not put file extension in name) reads from .csv
    string mlpPath = "/Users/tjohnsen/Documents/GitHub/MLP-Estimating-Exoplanet-Parameters/MLP_Results/5%_Noise/MLP_Epoch951";
    cout << "Enter .csv file path to Pretrained MLP (no file extension):" << endl;
    //cin >> mlpPath;
    
    // read pretrained MLP
    MLP* mlp = MLP::read(mlpPath);
    
    // fit data to output values using mlp
    mlp->predict(testDataSet, true);
    
    // path to output file (do not put file extension in name) outputs as .csv
    string outputPath = testDataPath.substr(0, testDataPath.size() - 4) + "_Results";
    
    // write data with predictions to output path
    testDataSet->write(outputPath);
    
    cout << "Results have been written to: " << outputPath << endl;
    
}

/*
 *** MAIN MAIN MAIN MAIN ***
 all logic and implmentation code starts here!
 *** MAIN MAIN MAIN MAIN ***
 */
int main() {
    //extractModels(vector<double>{0.3028, 0.99720, 0.0028}, true, "Albedo_Models_Repository", "", "Albedo_Models_248Bins0.3to1.0microns", ""); //0.5016, 0.99720, 0.0028 //0.3028, 0.99720, 0.0028
    //checkForDegeneracies("Albedo_Models_248Bins0.3to1.0microns", "");
    //dropDegeneraciesWithHighestFsed("Albedo_Models_248Bins0.3to1.0microns", "", true);
    //addBinaryCloudClass("Albedo_Models_Full_Resolution", "");
    //createDataSets("Albedo_Models_248Bins0.3to1.0microns", "_DroppedDegeneraciesFsed11", .8, .1, .1, vector<double>{10, .05, .1, .2});
    //trainMLP("Albedo_Models_248Bins0.3to1.0microns/MLP_Results/20% Noise");
    
    cout << "Enter 'train' to train new MLP or 'test' to test a pretrained MLP:" << endl;
    string choice = "";
    while(choice != "train" && choice != "test") {
        choice = readString();
        if(choice == "train")
            trainMLP_UI();
        else if (choice == "test")
            testMLP_UI();
        else
            cout << "invalid entry." << endl;
    }
    
    cout << "Press any key and return to exit." << endl;
    int ok;
    cin >> ok;
    return 0;
}

