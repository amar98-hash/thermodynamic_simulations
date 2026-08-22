#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>

#include <queue>
#include <array>
#include <algorithm>
#include <limits>
#include <stdexcept>


#include <aslGeomInc.h>
#include <aslDataInc.h>
#include <readers/aslVTKFormatReaders.h> //to read surface from STL file

#include <readers/aslVTKFormatReaders.h>
#include <writers/aslVTKFormatWriters.h>

#include <math/aslPositionFunction.h> //generatePFLinear
#include <math/aslDistanceFunction.h>   //dataInterpolator.
#include <num/aslBasicBC.h>  //for boundary conditions

#include <math/aslPositionFunction.h>
#include <utilities/aslParametersManager.h>
#include <math/aslTemplates.h>

#include <num/aslLBGK.h>
#include <num/aslLBGKBC.h>
#include <acl/DataTypes/aclMemBlock.h>

#include <utilities/aslParametersManager.h>
#include <acl/aclGenerators.h>

#include <num/aslFDAdvectionDiffusion.h>

#include <acl/aclMath/aclReductionAlgGenerator.h>
#include <acl/aclMath/aclVectorOfElementsOperations.h>

#include <num/aslFDStefanMaxwell.h> // for multicomponent StefanMaxwell.
#include "helper.h"

//boltsman constant
constexpr double kB = 1.380649e-23;   // J/K
// Universal gas constant
constexpr double R = 8.314462618;    // J/(mol K)
// Avrogadro's number
constexpr double NA = 6.02214076e23;   // mol^-1
//torr to Pa
const double torrToPa = 133.322368;

//nesmeyanov parameters for Li
//onstexpr double A1 = 4.98831; //cus C is used elsewhere.
//onstexpr double B1 = 7918.984;
//onstexpr double C1 = -9.52;

constexpr double A1 = 10.34540;
constexpr double B1 = 8345.574;
constexpr double C1 = -0.00008840;
constexpr double D1 = -0.68106;

//declaraation of the temperature function
struct WallTemperatureParameters
{
};

// ============================================================
// PRECOMPUTED SOURCE / SINK DATA
// ============================================================
struct CondensationCell
{
    int cIndex;          // index into cB
    double cSat;         // saturation concentration [mol/m^3]
};

struct SourceCell
{
    int cIndex;          // gas cell immediately above reservoir
    double cSat;         // imposed saturation concentration [mol/m^3]
    double temperature;  // for diagnostics only
};

struct SourceSinkCache
{
    std::vector<CondensationCell> condensationCells;
    std::vector<SourceCell> sourceCells;
};

struct ConcentrationCubeResult
{
    double minCAr = 0.0;
    double maxCAr = 0.0;
    double averageCAr = 0.0;

    double minCB = 0.0;
    double maxCB = 0.0;
    double averageCB = 0.0;

    double deltaCAr = 0.0;
    double deltaCB = 0.0;

    double relativeChangeCAr = 0.0;
    double relativeChangeCB = 0.0;

    std::size_t gasCells = 0;

    bool converged = false;
};


struct ConcentrationConvergenceState
{
    double previousAverageCAr = 0.0;
    double previousAverageCB = 0.0;

    bool initialized = false;
};


// Build once AFTER temperature reaches steady state
SourceSinkCache buildSourceSinkCache(const asl::SPDataWithGhostNodesACLData& cB,
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& SourceMap,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap);

// Very cheap version called during every SM iteration
void applySourceAndSinkFast(const asl::SPDataWithGhostNodesACLData& cB,const SourceSinkCache& cache);

asl::SPDataWithGhostNodesACLData buildGasVolumeMap(const asl::SPDataWithGhostNodesACLData& surface);
double calculatePeq(const asl::SPDataWithGhostNodesACLData& temperature,const asl::SPDataWithGhostNodesACLData& gasVolumeMap,double NAr);
void updatePeqField(const asl::SPDataWithGhostNodesACLData& PeqField,const asl::SPDataWithGhostNodesACLData& gasVolumeMap,double Peq);
double calculateTotalNAr(const asl::SPDataWithGhostNodesACLData& nAr,const asl::SPDataWithGhostNodesACLData& gasVolumeMap);
double sampleVaporPressure(double T, const double A, const double B, const double C, const double D);
void buildSourceMap(const asl::SPDataWithGhostNodesACLData& sourceMap,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double lengthZ,       // mm
    double thicknessY,    // mm
    double widthX,        // mm
    double zs,            // cuboid center z, mm
    double ys,            // cuboid center y, mm
    double xs);           // cuboid center x, mm

double calculateGasVolume(const asl::SPDataWithGhostNodesACLData& gasVolumeMap);

void initializeCAr(const asl::SPDataWithGhostNodesACLData& cAr,
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double PAr_Pa);

void updateNumberDensity(
    const asl::SPDataWithGhostNodesACLData& n,
    const asl::SPDataWithGhostNodesACLData& c,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap);

void updateMoleFractions(
    const asl::SPDataWithGhostNodesACLData& xLi,
    const asl::SPDataWithGhostNodesACLData& xAr,
    const asl::SPDataWithGhostNodesACLData& cLi,
    const asl::SPDataWithGhostNodesACLData& cAr,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap);

void applySourceAndSink(
    const asl::SPDataWithGhostNodesACLData& cLi,
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& SourceMap,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap);

void updatePressureFields(
    const asl::SPDataWithGhostNodesACLData& PAr,
    const asl::SPDataWithGhostNodesACLData& PB,
    const asl::SPDataWithGhostNodesACLData& Ptotal,
    const asl::SPDataWithGhostNodesACLData& xAr,
    const asl::SPDataWithGhostNodesACLData& xB,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double Pmix_Pa);

double calculatePeqBinary(
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& xAr,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double NAr);

void copyField(
    const asl::SPDataWithGhostNodesACLData& dst,
    const asl::SPDataWithGhostNodesACLData& src);

struct TemperatureCubeResult
{
    double minT;
    double averageT;
    std::size_t gasCells;
    bool converged=false;
};

TemperatureCubeResult checkTemperatureCube(
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double centerX,
    double centerY,
    double centerZ,
    double cubeLength_mm,
    double operatingT_K,
    double tolerance_K);


ConcentrationCubeResult checkConcentrationCube(
    const asl::SPDataWithGhostNodesACLData& cAr,
    const asl::SPDataWithGhostNodesACLData& cB,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,

    double centerX,
    double centerY,
    double centerZ,

    double cubeLength_mm,

    double relativeTolerance,
    double absoluteTolerance,

    ConcentrationConvergenceState& state);

typedef float FlT;

int main()
{
    double dx = 1.0; //step size in mm

    Parameters p;
    std::cout << "Reading heatpipe bounds..." << std::endl;
    std::string paramfile =  "C:/_dev/numerical_solvers/ASL-0.1.7/examples/heatpipe/heatpipe-parameters.txt";
    readKeyValues(paramfile, p);
    printKeyValues(p);
    const double zmax = p.zmax, xmax =  p.xmax, ymax = p.ymax;
    const double zmin = p.zmin, xmin =  p.xmin, ymin = p.ymin;
    const int zlen =  abs(zmax) +abs(zmin);
    const int ylen =  abs(ymax) +abs(ymin);
    const int xlen =  abs(xmax) +abs(xmin);
    const double zr =  p.zr, yr=p.yr, xr=p.xr;
    const double zcold =  p.zcold, ycold=p.ycold, xcold=p.xcold;

    std::cout<<"Creating computational block.."<<std::endl;                             
    asl::AVec<int> size =asl::makeAVec(zlen+40, ylen+40, xlen+40);//z,  y,    x
    asl::AVec<> origin =asl::makeAVec(zmin-20.0,ymin-20, xmin-20);      //z,  y,    x
    asl::Block initialBlock(size,dx,origin);
    std::cout << "Computational block created." << std::endl;

    std::cout << "Starting heatpipe STL import..." << std::endl;
    const std::string stlFile = "C:/_dev/numerical_solvers/ASL-0.1.7/examples/heatpipe/heatpipe-NaK.stl";
    auto heatpipe_cavity =asl::readSurface(stlFile,initialBlock);
    if (!heatpipe_cavity)
    {
        std::cerr<< "ERROR: Could not read STL."<< std::endl;
        return 1;
    }

    std::cout<< "STL read successfully." << std::endl;
    asl::Block block(heatpipe_cavity->getInternalBlock());
    std::cout<< "Internal block created." << std::endl;
    std::cout<< "Grid size: "<< block.getSize()[0] << " x "<< block.getSize()[1] << " x "<< block.getSize()[2]<< std::endl;

    //asl::AVec<int> size =asl::makeAVec(701, 81,151);
    //asl::AVec<> origin  =asl::makeAVec(-350.0,-40.0, -75.0);//-155.0);

    std::cout << "Reading Temperature boundary conditions..." << std::endl;
    double Tmax= 588.0+273.15;
    const double Tmin = 19.1+273.15; //fixed 19.1 C.
  

    // Experimental oven temperatures and pressures.
    std::cout << "Reading operating Temperature schemes..." << std::endl;
    std::vector<double> targetTemperatures_C = readTValues(paramfile);
    std::cout << "Reading operating Pressure schemes..." << std::endl;
    std::vector<double> targetPressure_Torr = readPValues(paramfile);

    //----------------------------------------------------------- 
    // ---------------------------------------------------------
    // Write ASL's STL-derived distance field
    // ---------------------------------------------------------
    std::cout << "Writing cavity geometry..." << std::endl;
    asl::WriterVTKXML writer("heatpipe_geometry_nak");
    writer.addScalars("cavity",*heatpipe_cavity);
    std::cout<< "Geometry written successfully as cavity-0."<< std::endl;

    std::cout << "Building gas Volume from cavity geometry..." << std::endl;
    auto gasVolumeMap = buildGasVolumeMap(heatpipe_cavity);
    writer.addScalars("gasVolumeMap", *gasVolumeMap);
    std::cout<< "Gas Volume map is generated from the Imported Geometry, saved as gasVolumeMap-0."<< std::endl;

    //with gas volumemap we can compute total voxel volume. ie, the volume of the heatpipe.
    const double V=calculateGasVolume(gasVolumeMap);
    const double P0_Torr = targetPressure_Torr[0]; //choose from pressure scheme.
    const double P0_Pa =  P0_Torr*torrToPa;
    std::cout<<"Calculated volume of the gas chamber :"<<V<<"m^3"<<std::endl;
    const double NAr = P0_Pa * V / (kB * Tmin);
    std::cout<<"Calculated NAr: "<<NAr<<std::endl;
    std::cout<<"NAr must remain constant throughout the simulation."<<std::endl;

    std::cout<<"Temperature scalar field on the surface of the heatpipe."<<std::endl;
    auto temperature =asl::generateDataContainerACL_SP<double>
    (
        block,   //3d region
        1,      // one scalar component
        1u      // ghost/border layer
    );

    //initialize the temperature field.
    std::cout<<"Initialize the temperature field to Tmin (19.1C)."<<std::endl;
    asl::initData(temperature,Tmin);

    std::cout<<"Now creating number density fields"<<std::endl;
    //density fields. B B2.. are names for generic samples. eg: Li, Li2 or NaK, Na, K, etc..
    auto nAr =asl::generateDataContainerACL_SP<double>(block,1,1u);
    auto nArRef =asl::generateDataContainerACL_SP<double>( block,1,1u);
    auto nB  = asl::generateDataContainerACL_SP<double>(block, 1, 1u);
    auto nB2 = asl::generateDataContainerACL_SP<double>(block, 1, 1u);

    std::cout<<"Now creating molar concentration fields"<<std::endl;
    //molar concentrations 
    auto cAr =asl::generateDataContainerACL_SP<double>( block,1,1u);
    auto cB =asl::generateDataContainerACL_SP<double>( block,1,1u);
    auto cB2 =asl::generateDataContainerACL_SP<double>( block,1,1u);
    std::cout<<"Now creating molar fraction fields."<<std::endl;
    auto xAr =asl::generateDataContainerACL_SP<double>(block, 1, 1u);
    auto xB =asl::generateDataContainerACL_SP<double>(block, 1, 1u);
    auto xB2 =asl::generateDataContainerACL_SP<double>(block, 1, 1u);
    std::cout<<"Now creating partial and total pressure fields.."<<std::endl;
    auto PArField =asl::generateDataContainerACL_SP<double>(block,1,1u);
    auto PBField =asl::generateDataContainerACL_SP<double>(block,1,1u);
    auto PtotalField =asl::generateDataContainerACL_SP<double>(block,1,1u);
   
    std::cout<<"Creating sample source map."<<std::endl;
    auto SourceMap =asl::generateDataContainerACL_SP<double>(block,1,1u);
    std::cout<<"Now creating total equillibrium pressure field.."<<std::endl;
    auto PeqField = asl::generateDataContainerACL_SP<double>(block,1,1u);

    std::cout<<"Initializing fields with 0.0s..."<<std::endl;
    asl::initData(xAr, 0.0); //it should be 1.0 initially but we'll update it later in simulation.
    asl::initData(xB, 0.0);
    asl::initData(nB, 0.0);
    asl::initData(nB2, 0.0);
    asl::initData(nAr, 0.0);
    asl::initData(nArRef, 0.0);
    asl::initData(cB, 0.0);
    asl::initData(cB2, 0.0);
    asl::initData(cAr, 0.0); //need to re-initialize cAr using Peq and T field at t=0.
    asl::initData(PArField,0.0);
    asl::initData(PBField,0.0);
    asl::initData(PtotalField,0.0);
    std::cout<<"B, B2.., Ar fields initialzied to uniform 0.0 values."<<std::endl;

    //boundary conditions.
    //std::vector<asl::SPNumMethod> thermalbc;
    const double PosZ   = zmax-zcold/2.0;
    const double PosX   = xmax-xcold/2.0;
    const double PosY   = ymax-ycold/2.0;

    //-------------------------------
    //-------------------------------

    auto coldZPlus  =asl::generateDFCylinder(zr+8.0,asl::makeAVec(zcold+5, 0.0, 0.0), asl::makeAVec(PosZ, 0.0, 0.0));   
    //auto coldZMinus =asl::generateDFCylinder(zr+1,asl::makeAVec(zcold+5, 0.0, 0.0),asl::makeAVec(-PosZ, 0.0, 0.0));
    auto coldXPlus  =asl::generateDFCylinder(xr+8.0,asl::makeAVec(0.0, 0.0, xcold+5), asl::makeAVec(0.0, 0.0, PosX));
    auto coldXMinus =asl::generateDFCylinder(xr+8.0,asl::makeAVec(0.0, 0.0, xcold+5),asl::makeAVec(0.0, 0.0, -PosX));
    auto coldYPlus  =asl::generateDFCylinder(yr+8.0,asl::makeAVec(0.0, ycold+5, 0.0), asl::makeAVec(0.0,PosY, 0.0));    
    auto coldYMinus =asl::generateDFCylinder(yr+8.0,asl::makeAVec(0.0, ycold+5, 0.0),asl::makeAVec(0.0,-PosY, 0.0));

    //Turn stl discrete data into continuous distance function.
    auto cavityDF =std::make_shared<asl::DataInterpolation>(heatpipe_cavity); 
    auto solidRegion =-cavityDF;
    
    //auto coldRegion =zPlusPlane| zMinusPlane| xPlusPlane| xMinusPlane;
    auto coldRegion =coldZPlus | coldXMinus|coldXPlus | coldYMinus|coldYPlus;
    auto coldWall =  cavityDF & coldRegion;
    auto hotWall  =  cavityDF & (-coldRegion);


    auto coldSolid =solidRegion & coldRegion;
    auto hotSolid =solidRegion & (-coldRegion);
    auto coldWallMap =asl::generateDataContainerACL_SP<FlT>(block,1,1u);
    auto hotWallMap = asl::generateDataContainerACL_SP<FlT>(block,1,1u);

    //initialize boundary maps
    std::cout<<"Initializing different heatpipe walls+windows with correct boundary condition temperatures"<<std::endl;
    asl::initData(coldWallMap,asl::normalize(coldWall, dx));
    asl::initData(hotWallMap,asl::normalize(hotWall, dx));

    auto solidMap =asl::generateDataContainerACL_SP<FlT>( block,1,1u);
    auto coldSolidMap =asl::generateDataContainerACL_SP<FlT>(block,1,1u);
    auto hotSolidMap =asl::generateDataContainerACL_SP<FlT>(block,1,1u);

    // Initialize volumetric masks
    asl::initData(solidMap,asl::normalize(solidRegion,dx));
    asl::initData(coldSolidMap,asl::normalize(coldSolid,dx));
    asl::initData(hotSolidMap,asl::normalize(hotSolid,dx));



    //imposing boundary conditins.
    const double T = targetTemperatures_C[0]+273.15;
    //thermalbc.push_back(asl::generateBCConstantValue(temperature,T,hotWallMap));
    //thermalbc.push_back(asl::generateBCConstantValue(temperature,Tmin,coldWallMap));
    //asl::initAll(thermalbc);

    //now build Li/Li2 source map.
    std::cout<<"Creating a sample source of about 10x1x10 mm^3 cuboid at the bottom of the center of the heatpipe."<<std::endl;
    buildSourceMap(SourceMap,gasVolumeMap,
        p.sample_lz,      // length along z, mm
        p.sample_ly,       // thickness along y, mm
        p.sample_lx,       // width along x, mm
        p.sample_z,       // z center
        p.sample_y,       // y center near bottom wall
        p.sample_x        // x center
    );

    //paraview debug
    writer.addScalars("coldWallMap", *coldWallMap);
    writer.addScalars("hotWallMap",  *hotWallMap);
    writer.addScalars("coldSolidMap", *coldSolidMap);
    writer.addScalars("hotSolidMap",  *hotSolidMap);

 
    //Temperature diffusion solver
    auto templ = &asl::d3q15();
    double dt = 0.01;        // s
    double alpha = 10.0;     // mm^2/s, temporary test value
    double diffCoefNum =alpha * dt / (dx * dx);
    auto nm =generateFDAdvectionDiffusion(temperature,diffCoefNum,templ);
    nm->init();

    // ============================================================
    // BINARY B-Ar STEFAN-MAXWELL SOLVER
    // ============================================================
    double D_BAr = 0.03492;      // m^2/s, 
    double dtB   = 2.5e-6;      // s
    double dx_m =block.dx * 1.0e-3;
    double Dnum =D_BAr * dtB /(dx_m * dx_m);
    std::cout<< "The required stable Dnum is "<<Dnum<<std::endl;
    auto templ2 =&asl::d3q7();
    auto smTransport =asl::generateFDStefanMaxwell(cB,cAr,Dnum,templ2);

    double DdustNum =0.5 * Dnum; //set the large pre-factor to Ddust to make D_eff ~ D_BAr. 
    smTransport->setDustDiffusionCoefficient(0,acl::generateVEConstant(DdustNum));
    smTransport->setDustDiffusionCoefficient(1,acl::generateVEConstant(DdustNum));
    smTransport->init();

    // ============================================================
    // SPECIES WALL BOUNDARY CONDITIONS
    // ============================================================
    std::cout<<"Impose impermeable boundary condition to the gases."<<std::endl;
    std::vector<asl::SPNumMethod> smBC;
    // B cannot pass through steel wall
    smBC.push_back(asl::generateBCConstantGradient2(cB,0.0,heatpipe_cavity,templ2));
    // Ar cannot pass through steel wall
    smBC.push_back(asl::generateBCConstantGradient2(cAr,0.0,heatpipe_cavity,templ2));
    initAll(smBC);
    
    writer.addScalars("temperature", *temperature);
    writer.addScalars("Peq_Torr", *PeqField);
    writer.addScalars("nAr", *nAr);
    writer.addScalars("SourceMap", *SourceMap);
    writer.addScalars("nB", *nB);
    writer.addScalars("cB", *cB);
    writer.addScalars("cAr", *cAr);
    writer.addScalars("xB", *xB);
    writer.addScalars("xAr", *xAr);
    writer.addScalars("PAr_Torr", *PArField);
    writer.addScalars("PB_Torr", *PBField);
    writer.addScalars("Ptotal_Torr", *PtotalField);
    writer.addScalars("nArRef",*nArRef);

    // ============================================================
    // QUASI-STATIC TEMPERATURE CASES
    // ============================================================
    // Number of numerical iterations.
    // These DO NOT represent physical seconds.
    // We are simply advancing the fields toward steady state.
    const unsigned int thermalSteps = 200000;//something large.. when the tempr converge it will breakup the loop anyway.
    const unsigned int smSteps      = 2000000;

    //executeAll(thermalbc);
    // ============================================================
    // LOOP THROUGH TEMPERATURE CASES
    // ============================================================
    for (std::size_t caseID = 0;caseID < targetTemperatures_C.size();++caseID)
    {
        // Current hot-wall temperature
        const double Thot_C =targetTemperatures_C[caseID];
        const double Thot_K =Thot_C + 273.15;

        std::cout
            << "\n========================================\n"
            << "Temperature case " << caseID + 1
            << "\nHot wall = " << Thot_C << " C"
            << "\nHot wall = " << Thot_K << " K"
            << "\n========================================\n";
            

        // REInitialize TEMPERATURE BOUNDARY CONDITIONS
        std::vector<asl::SPNumMethod> thermalbc;
        thermalbc.push_back(asl::generateBCConstantValue(temperature,Tmin,coldWallMap));
        thermalbc.push_back(asl::generateBCConstantValue(temperature,Thot_K,hotWallMap));
        initAll(thermalbc);
       
        // EVOLVE TEMPERATURE TO STEADY STATE
        std::cout << "Executing temperature diffusion...\n";
        for (unsigned int iteration = 1;iteration <= thermalSteps;++iteration)
        {
            auto tmstart = std::chrono::high_resolution_clock::now();
            nm->execute();
            executeAll(thermalbc);
            
            TemperatureCubeResult result;
            if (!(iteration % 500))
            {

                //writer.write();
                result =checkTemperatureCube(temperature,gasVolumeMap,0,0,0, 20, Thot_K,1.0);
                auto tmend = std::chrono::high_resolution_clock::now();
                std::cout<< "  thermal iteration "<< iteration<< " / "<< thermalSteps<< std::endl;
                std::chrono::duration<double> elapsed = tmend - tmstart;
                std::cout<< "Thermal diffusion 100th step Elapsed time = "<< elapsed.count()<< " seconds"<< std::endl;
            }
           
            if(result.converged ==true) {break; std::cout<<"Temperature converged.. exiting thermal diffusion"<<std::endl;}
        }
       
        std::cout<< "Steady-state Temperature field computation complete.\n";
        //return 0;    

        // RESET B FOR THIS TEMPERATURE CASE
        // This makes every temperature case independent.
        asl::initData(cB, 0.0);
        asl::initData(nB, 0.0);
        asl::initData(xAr,0.0);
        asl::initData(xB,0.0);
        asl::initData(PeqField,0.0);

        //CALCULATE QUASI-STATIC, UNIFORM INITIAL ARGON PRESSURE 
        double Peq_Pa =calculatePeq(temperature, gasVolumeMap,NAr);
        const double Peq_Torr =Peq_Pa / 133.322368;
        std::cout<< "Equillibirum Pressure at "<<Thot_C<<" : "<< Peq_Torr<< " Torr\n";
        
        auto start1 = std::chrono::high_resolution_clock::now();
        updatePeqField(PeqField,gasVolumeMap,Peq_Torr);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed1 = end1 - start1;
        std::cout<< "Elapsed time = "<< elapsed1.count()<< " seconds"<< std::endl;

        initializeCAr(cAr, temperature, gasVolumeMap, Peq_Pa);
        updateNumberDensity(nAr,cAr,gasVolumeMap);
        copyField(nArRef,nAr);
        updateMoleFractions( xB,xAr,cB,cAr,gasVolumeMap);
        updatePressureFields(PArField,PBField,PtotalField,xAr,xB,gasVolumeMap, Peq_Pa);
   
        // CHECK ARGON CONSERVATION
        const double NAr_check =calculateTotalNAr(nAr,gasVolumeMap);
        std::cout<< "NAr initial = "<< NAr<< "\nNAr check   = "
        << NAr_check<< "\nrelative error = "<< std::abs(NAr_check - NAr) / NAr<< std::endl;

        // ESTABLISH SATURATED B SOURCE
        std::cout<< "Precomputing B source/sink cells for this temperature..."<< std::endl;

        const SourceSinkCache sourceSinkCache =buildSourceSinkCache(cB,temperature,SourceMap,gasVolumeMap);

        std::cout<< "Source/sink cache ready."<< std::endl;
        applySourceAndSinkFast(cB,sourceSinkCache);
        //applySourceAndSink(cB,temperature,SourceMap,gasVolumeMap);
       
        executeAll(smBC);       //No species transport through steel walls

        double Pmix_Pa =Peq_Pa;
      
        ConcentrationConvergenceState concentrationState;

        //Ar-B TRANSPORT USING STEFAN-MAXWELL'S DIFFUSION 
        std::cout << "Evolving Ar-B diffusion...\n";
        const auto smStart =std::chrono::high_resolution_clock::now();
        for (unsigned int iteration = 1;iteration <= smSteps;++iteration)
        {
            //const auto smStart =std::chrono::high_resolution_clock::now();
            smTransport->execute(); // Coupled Li-Ar Stefan-Maxwell update
            executeAll(smBC);       // No species transport through steel walls
            //const auto smEnd =std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double> smElapsed = smEnd - smStart;
            //std::cout<< "sm Elapsed time = "<< smElapsed.count()<< " seconds"<< std::endl;
            // B reservoir restores saturated composition
            applySourceAndSinkFast(cB,sourceSinkCache);
            //applySourceAndSink(cB,temperature,SourceMap,gasVolumeMap);
        
            if (!(iteration % 500))
            {
                const auto cResult =
                checkConcentrationCube(cAr,cB,gasVolumeMap,0,0,0,20, 1.0e-5,  1.0e-12,concentrationState);
                //const auto smStart1 =std::chrono::high_resolution_clock::now();
                updateNumberDensity(nB,cB,gasVolumeMap);
                updateNumberDensity(nAr,cAr,gasVolumeMap);
                updateMoleFractions(xB,xAr,cB,cAr,gasVolumeMap);
                double Pmix_Pa =calculatePeqBinary(temperature,xAr,gasVolumeMap,NAr);
                updatePressureFields(PArField,PBField,PtotalField,xAr,xB,gasVolumeMap, Pmix_Pa);
                updatePeqField(PeqField,gasVolumeMap,Pmix_Pa / torrToPa);
                writer.write();
                std::cout<< "  STEFAN-MAXWELL iteration "<< iteration<< " / "<< smSteps<< " | Pmix = "<< Pmix_Pa / torrToPa<< " Torr"<<std::endl;
                //const auto smEnd1 =std::chrono::high_resolution_clock::now();
                //std::chrono::duration<double> smElapsed1 = smEnd1 - smStart1;
                //std::cout<< "sm Elapsed1 time = "<< smElapsed1.count()<< " seconds"<< std::endl;
                if(cResult.converged == true) break;
            }
        }
        const auto smEnd =std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> smElapsed = smEnd - smStart;
        std::cout<< "sm diffusion Elapsed time = "<< smElapsed.count()<< " seconds"<< std::endl;

        std::cout<< "Ar-B diffusion computation complete...\n";
        //updateNumberDensity(nB,cB,gasVolumeMap);
        //updateNumberDensity(nAr,cAr,gasVolumeMap);
        //updateMoleFractions(xB,xAr,cB,cAr,gasVolumeMap);
        //writer.write();

        // CHECK FINAL ARGON CONSERVATION
        const double NAr_final =calculateTotalNAr(nAr,gasVolumeMap);
        const double ArRelativeError =std::abs(NAr_final - NAr) / NAr;
        
        std::cout<< "Final total pressure = "
            << Peq_Pa / 133.322368<< " Torr\n"
            << "NAr target = "<< NAr
            << "\nNAr final  = "<< NAr_final
            << "\nAr conservation relative error = "
            << ArRelativeError<< std::endl;
        std::cout<< "Finished diffusion for "<< Thot_C << " C case.\n";
    }
    std::cout<< "=== Finished ==="<< std::endl;
    return 0;
}

void updatePressureFields(
    const asl::SPDataWithGhostNodesACLData& PAr,
    const asl::SPDataWithGhostNodesACLData& PB,
    const asl::SPDataWithGhostNodesACLData& Ptotal,
    const asl::SPDataWithGhostNodesACLData& xAr,
    const asl::SPDataWithGhostNodesACLData& xB,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double Pmix_Pa)
{
    // ------------------------------------------------------------
    // Internal computational region
    // ------------------------------------------------------------
    const auto& internal =xAr->getInternalBlock();
    const auto size =internal.getSize();
    // ------------------------------------------------------------
    // Blocks
    // ------------------------------------------------------------
    const auto& bPA =PAr->getBlock();
    const auto& bPB =PB->getBlock();
    const auto& bPT =Ptotal->getBlock();
    const auto& bXA =xAr->getBlock();
    const auto& bXB =xB->getBlock();
    const auto& bG = gasVolumeMap->getBlock();


    // ------------------------------------------------------------
    // Ghost widths
    // ------------------------------------------------------------
    const int gPA =PAr->getGhostBorder();
    const int gPB =PB->getGhostBorder();
    const int gPT =Ptotal->getGhostBorder();
    const int gXA =xAr->getGhostBorder();
    const int gXB =xB->getGhostBorder();
    const int gG =gasVolumeMap->getGhostBorder();

    // ------------------------------------------------------------
    // Map fields
    // ------------------------------------------------------------
    auto paMap =acl::map<double>(PAr->getContainer()[0]);
    auto pbMap =acl::map<double>(PB->getContainer()[0]);
    auto ptMap =acl::map<double>(Ptotal->getContainer()[0]);
    auto xaMap =acl::map<double>(xAr->getContainer()[0]);
    auto xbMap =acl::map<double>(xB->getContainer()[0]);
    auto gMap =acl::map<double>(gasVolumeMap->getContainer()[0]);
    double* PA =paMap.get();
    double* PBv =pbMap.get();
    double* PT = ptMap.get();
    const double* XA =xaMap.get();
    const double* XB = xbMap.get();
    const double* G =gMap.get();


    // ------------------------------------------------------------
    // Uniform equilibrium mixture pressure
    // ------------------------------------------------------------
    const double Pmix_Torr = Pmix_Pa / torrToPa;
    constexpr double eps =1.0e-30;

    // ============================================================
    // Build pressure fields
    //
    //      PAr    = xAr * Pmix
    //      PB     = xB  * Pmix
    //      Ptotal = Pmix
    //
    // ============================================================

    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idPA =bPA.c2i(asl::makeAVec(i + gPA,j + gPA, k + gPA));
        const int idPB = bPB.c2i( asl::makeAVec(i + gPB,j + gPB,k + gPB) );
        const int idPT =bPT.c2i(asl::makeAVec(i + gPT,j + gPT,k + gPT));
        const int idXA =bXA.c2i( asl::makeAVec(i + gXA,j + gXA,k + gXA));
        const int idXB = bXB.c2i( asl::makeAVec(i + gXB,j + gXB,k + gXB));
        const int idG =bG.c2i(asl::makeAVec(i + gG,j + gG,k + gG));

        // --------------------------------------------------------
        // Inside gas cavity
        // --------------------------------------------------------
        if (G[idG] > 0.5)
        {
            double xa =std::max(0.0,XA[idXA]);
            double xb =std::max(0.0,XB[idXB]);

            // ----------------------------------------------------
            // Normalize defensively.
            //
            // updateMoleFractions() should already make:
            //
            //      xa + xb = 1
            //
            // but this removes tiny numerical errors.
            // ----------------------------------------------------
            const double xsum = xa + xb;

            if (xsum > eps)
            {
                xa /= xsum;
                xb /= xsum;
            }
            else
            {
                xa = 0.0;
                xb = 0.0;
            }


            // ----------------------------------------------------
            // Dalton's law
            // ----------------------------------------------------
            PA[idPA] =xa * Pmix_Torr;
            PBv[idPB] =xb * Pmix_Torr;

            // ----------------------------------------------------
            // Mechanical-equilibrium total pressure
            // ----------------------------------------------------
            PT[idPT] =
                Pmix_Torr;
        }

        // --------------------------------------------------------
        // Solid / outside chamber
        // --------------------------------------------------------
        else
        {
            PA[idPA]  = 0.0;
            PBv[idPB] = 0.0;
            PT[idPT]  = 0.0;
        }
    }
}

void updateNumberDensity(
    const asl::SPDataWithGhostNodesACLData& n,
    const asl::SPDataWithGhostNodesACLData& c,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap)
{
    // ------------------------------------------------------------
    // Avogadro constant
    //
    // c : molar concentration [mol/m^3]
    // n : number density       [particles/m^3]
    //
    // n = N_A * c
    // ------------------------------------------------------------

    // ------------------------------------------------------------
    // Internal computational block
    // ------------------------------------------------------------
    const auto& internal =n->getInternalBlock();

    // ------------------------------------------------------------
    // ASL blocks
    // ------------------------------------------------------------
    const auto& bN =n->getBlock();
    const auto& bC =c->getBlock();
    const auto& bM =gasVolumeMap->getBlock();
    const auto size =internal.getSize();

    // ------------------------------------------------------------
    // Ghost-border widths
    // ------------------------------------------------------------
    const int gn =n->getGhostBorder();
    const int gc =c->getGhostBorder();
    const int gm =gasVolumeMap->getGhostBorder();


    // ------------------------------------------------------------
    // Map ASL fields to host memory
    // ------------------------------------------------------------
    auto nMap =acl::map<double>(n->getContainer()[0]);
    auto cMap =acl::map<double>(c->getContainer()[0]);
    auto mMap =acl::map<double>(gasVolumeMap->getContainer()[0]);

    double* N =nMap.get();

    const double* C =cMap.get();
    const double* M =mMap.get();


    // ------------------------------------------------------------
    // Loop through computational domain
    // ------------------------------------------------------------
    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idN =bN.c2i(asl::makeAVec(i + gn,j + gn,k + gn));
        const int idC =bC.c2i(asl::makeAVec(i + gc,j + gc,k + gc));
        const int idM =bM.c2i(asl::makeAVec(i + gm,j + gm,k + gm));


        // --------------------------------------------------------
        // Gas voxel
        // --------------------------------------------------------
        if (M[idM] > 0.5)
        {
            N[idN] =NA * C[idC];
        }
        // --------------------------------------------------------
        // Wall / outside heat-pipe cavity
        // --------------------------------------------------------
        else
        {
            N[idN] = 0.0;
        }
    }
}
void initializeCAr(
    const asl::SPDataWithGhostNodesACLData& cAr,
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double PAr_Pa)
{
    // ------------------------------------------------------------
    // Internal computational block
    // ------------------------------------------------------------
    const auto& internal = cAr->getInternalBlock();

    // ------------------------------------------------------------
    // ASL blocks
    // ------------------------------------------------------------
    const auto& bC = cAr->getBlock();
    const auto& bT = temperature->getBlock();
    const auto& bM = gasVolumeMap->getBlock();

    const auto size = internal.getSize();

    // ------------------------------------------------------------
    // Ghost-border widths
    // ------------------------------------------------------------
    const int gc = cAr->getGhostBorder();
    const int gt = temperature->getGhostBorder();
    const int gm = gasVolumeMap->getGhostBorder();

    // ------------------------------------------------------------
    // Map ASL data to host memory
    // ------------------------------------------------------------
    auto cMap =acl::map<double>(cAr->getContainer()[0]);

    auto tMap =acl::map<double>(temperature->getContainer()[0]);

    auto mMap =acl::map<double>(gasVolumeMap->getContainer()[0]);

    double* C = cMap.get();

    const double* T = tMap.get();
    const double* M = mMap.get();

    // ------------------------------------------------------------
    // Loop through computational gas volume
    // ------------------------------------------------------------
    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idC =bC.c2i(asl::makeAVec(i + gc,j + gc,k + gc));
        const int idT =bT.c2i(asl::makeAVec(i + gt,j + gt,k + gt));
        const int idM =bM.c2i(asl::makeAVec(i + gm,j + gm,k + gm));
        // --------------------------------------------------------
        // Gas voxel
        //
        // Ideal gas:
        //
        //      P_Ar = c_Ar R T
        //
        // therefore:
        //
        //      c_Ar = P_Ar / (R T)
        //
        // Units:
        //
        // P  : Pa
        // R  : J/(mol K)
        // T  : K
        // c  : mol/m^3
        // --------------------------------------------------------
        if (M[idM] > 0.5)
        {
            C[idC] =PAr_Pa /(R * T[idT]);
        }
        else
        {
            // Wall / outside of gas cavity
            C[idC] = 0.0;
        }
    }
}

double calculatePeqBinary(
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& xAr,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double NAr)
{

    const auto& internal =temperature->getInternalBlock();
    const auto& bT =temperature->getBlock();
    const auto& bX =xAr->getBlock();
    const auto& bM =gasVolumeMap->getBlock();
    const auto size =internal.getSize();
    const int gt =temperature->getGhostBorder();
    const int gx =xAr->getGhostBorder();
    const int gm =gasVolumeMap->getGhostBorder();

    auto tMap = acl::map<double>(temperature->getContainer()[0]);
    auto xMap = acl::map<double>(xAr->getContainer()[0]);
    auto mMap =acl::map<double>(gasVolumeMap->getContainer()[0]);

    const double* T =tMap.get();
    const double* X =xMap.get();
    const double* M =mMap.get();
    const double dx_m =internal.dx * 1e-3;
    const double dV =dx_m * dx_m * dx_m;
    double integral = 0.0;

    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idT =bT.c2i(asl::makeAVec(i + gt,j + gt,k + gt));
        const int idX =bX.c2i(asl::makeAVec(i + gx,j + gx,k + gx));
        const int idM =bM.c2i(asl::makeAVec(i + gm,j + gm,k + gm));

        if (M[idM] > 0.5)
        {
            integral +=X[idX] *dV /T[idT];
        }
    }

    return NAr * kB /integral;
}

void updateMoleFractions(
    const asl::SPDataWithGhostNodesACLData& xLi,
    const asl::SPDataWithGhostNodesACLData& xAr,
    const asl::SPDataWithGhostNodesACLData& cLi,
    const asl::SPDataWithGhostNodesACLData& cAr,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap)
{
    // ------------------------------------------------------------
    // Internal computational block
    // ------------------------------------------------------------
    const auto& internal =cLi->getInternalBlock();

    // ------------------------------------------------------------
    // ASL blocks
    // ------------------------------------------------------------
    const auto& bXL =xLi->getBlock();
    const auto& bXA =xAr->getBlock();
    const auto& bCL =cLi->getBlock();
    const auto& bCA =cAr->getBlock();
    const auto& bM =gasVolumeMap->getBlock();
    const auto size =internal.getSize();

    // ------------------------------------------------------------
    // Ghost-border widths
    // ------------------------------------------------------------
    const int gXL =xLi->getGhostBorder();
    const int gXA =xAr->getGhostBorder();
    const int gCL =cLi->getGhostBorder();
    const int gCA =cAr->getGhostBorder();
    const int gM =gasVolumeMap->getGhostBorder();

    // ------------------------------------------------------------
    // Map ASL fields to host memory
    // ------------------------------------------------------------
    auto xLiMap =acl::map<double>(xLi->getContainer()[0]);
    auto xArMap =acl::map<double>(xAr->getContainer()[0]);
    auto cLiMap =acl::map<double>(cLi->getContainer()[0]);
    auto cArMap =acl::map<double>(cAr->getContainer()[0]);
    auto mMap =acl::map<double>(gasVolumeMap->getContainer()[0]);
    double* XL =xLiMap.get();
    double* XA =xArMap.get();
    const double* CL =cLiMap.get();
    const double* CA =cArMap.get();
    const double* M =mMap.get();

    // Small number to avoid division by zero
    constexpr double eps = 1.0e-30;

    // ------------------------------------------------------------
    // Loop through computational domain
    // ------------------------------------------------------------
    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idXL =bXL.c2i(asl::makeAVec(i + gXL,j + gXL, k + gXL));
        const int idXA =bXA.c2i(asl::makeAVec(i + gXA,j + gXA,k + gXA));
        const int idCL =bCL.c2i(asl::makeAVec(i + gCL,j + gCL,k + gCL));
        const int idCA =bCA.c2i(asl::makeAVec(i + gCA,j + gCA,k + gCA));
        const int idM =bM.c2i(asl::makeAVec(i + gM,j + gM,k + gM));

        // --------------------------------------------------------
        // Inside gas cavity
        // --------------------------------------------------------
        if (M[idM] > 0.5)
        {
            const double cLiLocal =std::max(0.0, CL[idCL]);
            const double cArLocal =std::max(0.0, CA[idCA]);
            const double cTotal =cLiLocal + cArLocal;

            if (cTotal > eps)
            {
                XL[idXL] =cLiLocal / cTotal;
                XA[idXA] =cArLocal / cTotal;
            }
            else
            {
                // No gas present
                XL[idXL] = 0.0;
                XA[idXA] = 0.0;
            }
        }

        // --------------------------------------------------------
        // Wall / outside gas cavity
        // --------------------------------------------------------
        else
        {
            XL[idXL] = 0.0;
            XA[idXA] = 0.0;
        }
    }
}


void buildSourceMap(
    const asl::SPDataWithGhostNodesACLData& sourceMap,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double lengthZ,       // mm
    double thicknessY,    // mm
    double widthX,        // mm
    double zs,            // cuboid center z, mm
    double ys,            // cuboid center y, mm
    double xs)            // cuboid center x, mm
{
    const auto& internal = sourceMap->getInternalBlock();
    const auto& blockS = sourceMap->getBlock();
    const auto& blockM = gasVolumeMap->getBlock();

    const auto size = internal.getSize();
    const int gs = sourceMap->getGhostBorder();
    const int gm = gasVolumeMap->getGhostBorder();

    // Map ASL fields to host memory
    auto sourceMapped =acl::map<double>(sourceMap->getContainer()[0]);
    auto maskMapped =acl::map<double>(gasVolumeMap->getContainer()[0]);

    double* S = sourceMapped.get();
    const double* M = maskMapped.get();

    // ---------------------------------
    // Initialize entire source map to 0
    // ---------------------------------

    const auto fullSize = blockS.getSize();

    const std::size_t nTotal =
        static_cast<std::size_t>(fullSize[0]) *
        static_cast<std::size_t>(fullSize[1]) *
        static_cast<std::size_t>(fullSize[2]);

    std::fill(S, S + nTotal, 0.0);

    // Half dimensions
    const double hz = 0.5 * lengthZ;
    const double hy = 0.5 * thicknessY;
    const double hx = 0.5 * widthX;

    // ---------------------------------
    // Construct cuboid source
    // ---------------------------------
    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idS = blockS.c2i(asl::makeAVec(i + gs,j + gs,k + gs));
        const int idM = blockM.c2i(asl::makeAVec(i + gm,j + gm,k + gm));

        // Your coordinate convention:
        // component 0 = z
        // component 1 = y
        // component 2 = x
        const double z =internal.position[0] + i * internal.dx;
        const double y =internal.position[1] + j * internal.dx;
        const double x =internal.position[2] + k * internal.dx;

        // Is this voxel inside our cuboid?
        const bool insideCuboid =
            (std::abs(z - zs) <= hz) &&
            (std::abs(y - ys) <= hy) &&
            (std::abs(x - xs) <= hx);

        // Only allow source voxels inside gas cavity
        if (insideCuboid && M[idM] > 0.5)
        {
            S[idS] = 1.0;
        }
    }
}

void applySourceAndSink(
    const asl::SPDataWithGhostNodesACLData& cB,
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& SourceMap,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap)
{

    // ------------------------------------------------------------
    // Internal computational region
    // ------------------------------------------------------------
    const auto& internal = cB->getInternalBlock();
    const auto size =internal.getSize();

    // ------------------------------------------------------------
    // ASL blocks
    // ------------------------------------------------------------
    const auto& bC =cB->getBlock();
    const auto& bT =temperature->getBlock();
    const auto& bS =SourceMap->getBlock();
    const auto& bG =gasVolumeMap->getBlock();


    // ------------------------------------------------------------
    // Ghost border widths
    // ------------------------------------------------------------
    const int gc =cB->getGhostBorder();
    const int gt =temperature->getGhostBorder();
    const int gs =SourceMap->getGhostBorder();
    const int gg =gasVolumeMap->getGhostBorder();


    // ------------------------------------------------------------
    // Map fields to host memory
    // ------------------------------------------------------------
    auto cBMap =acl::map<double>( cB->getContainer()[0]);
    auto temperatureMap =acl::map<double>(temperature->getContainer()[0]);
    auto sourceMap =acl::map<double>(SourceMap->getContainer()[0]);
    auto gasMap =acl::map<double>(gasVolumeMap->getContainer()[0]);

    double* C =cBMap.get();
    const double* T =temperatureMap.get();
    const double* S =sourceMap.get();
    const double* G =gasMap.get();


    // ============================================================
    // PART 1
    //
    // CONDENSATION SINK
    //
    // Any gas voxel touching a solid/wall voxel is checked.
    //
    // If
    //
    //      cB > cB_sat(T)
    //
    // excess gaseous B is condensed out.
    // ============================================================

    double totalRemovedConcentration = 0.0;
    unsigned int condensationCells = 0;


    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idC =bC.c2i(asl::makeAVec(i + gc,j + gc,k + gc));
        const int idT =bT.c2i(asl::makeAVec(i + gt,j + gt,k + gt));
        const int idG =bG.c2i(asl::makeAVec(i + gg,j + gg,k + gg));

        // Only gas cells
        if (G[idG] <= 0.5)
            continue;

        // --------------------------------------------------------
        // Check six nearest neighbours.
        // If any neighbour is not gas, this gas voxel lies
        // immediately next to a wall.
        // --------------------------------------------------------

        const int idG_ip =bG.c2i(asl::makeAVec(i + 1 + gg,j+ gg,k+ gg));
        const int idG_im =bG.c2i(asl::makeAVec(i - 1 + gg,j + gg,k+ gg));
        const int idG_jp =bG.c2i(asl::makeAVec(i + gg,j + 1 + gg,k     + gg));
        const int idG_jm =bG.c2i(asl::makeAVec(i + gg,j - 1 + gg,k + gg));
        const int idG_kp=bG.c2i(asl::makeAVec(i + gg,j + gg, k + 1 + gg));
        const int idG_km =bG.c2i(asl::makeAVec(i + gg,j + gg,k - 1 + gg));

        const bool adjacentToWall =
            (G[idG_ip] <= 0.5) ||
            (G[idG_im] <= 0.5) ||
            (G[idG_jp] <= 0.5) ||
            (G[idG_jm] <= 0.5) ||
            (G[idG_kp] <= 0.5) ||
            (G[idG_km] <= 0.5);

        if (!adjacentToWall)
            continue;

        const double Tw =T[idT];
        if (Tw <= 0.0)continue;

        // --------------------------------------------------------
        // Local B sample saturation pressure from Nesmeyanov curve
        // --------------------------------------------------------
        const double PB_sat =sampleVaporPressure(Tw,A1,B1,C1, D1);     // Pa

        // --------------------------------------------------------
        // Saturated molar concentration
        //
        // c = P / (R T)
        // --------------------------------------------------------
        const double cB_sat =PB_sat / (R * Tw);


        // --------------------------------------------------------
        // Condense only the supersaturated amount
        // --------------------------------------------------------
        if (C[idC] > cB_sat)
        {
            totalRemovedConcentration +=C[idC] - cB_sat;
            C[idC] =cB_sat;
            ++condensationCells;
        }
    }


    // ============================================================
    // PART 2
    //
    // Li RESERVOIR SOURCE
    //
    // Find the top surface of the SourceMap and impose
    //
    //       cLi = Psat(Ts)/(R Ts)
    //
    // in the gas cell immediately above it.
    //
    // This is done AFTER condensation so the reservoir boundary
    // always wins.
    // ============================================================

    double maxSourceTemperature = 0.0;
    double maxSourceConcentration = 0.0;
    double maxSourceDelta = 0.0;

    unsigned int sourceCellCount = 0;


    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idS =bS.c2i(asl::makeAVec(i + gs,j + gs,k + gs));
        // Not part of Li reservoir
        if (S[idS] <= 0.5) continue;

        // --------------------------------------------------------
        // Cell immediately above source (+y = +j)
        // --------------------------------------------------------
        const int idSAbove =bS.c2i(asl::makeAVec(i + gs+1,j  + gs,k + gs));

        // If source exists above, this is not the top surface.
        if (S[idSAbove] > 0.5) continue;
        const int idGAbove =bG.c2i(asl::makeAVec(i + gg+1,j  + gg,k + gg));
        // Gas must exist immediately above source
        if (G[idGAbove] <= 0.5) continue;

        const int idCAbove =bC.c2i(asl::makeAVec(i + gc+1,j  + gc,k  + gc));
        const int idTAbove =bT.c2i(asl::makeAVec(i  + gt+1,j  + gt,k + gt));

        const double Ts =T[idTAbove];

        if (Ts <= 0.0) continue;

        // --------------------------------------------------------
        // Saturation pressure over liquid B
        // --------------------------------------------------------
        const double PB_sat =sampleVaporPressure(Ts, A1, B1, C1, D1);

        // --------------------------------------------------------
        // Saturated B concentration
        // --------------------------------------------------------
        const double cB_source =PB_sat /(R * Ts);

        // --------------------------------------------------------
        // How much B must the reservoir replenish?
        // --------------------------------------------------------
        const double delta =cB_source -C[idCAbove];

        maxSourceDelta =std::max(maxSourceDelta,std::abs(delta));
        maxSourceTemperature =std::max(maxSourceTemperature, Ts);
        maxSourceConcentration =std::max( maxSourceConcentration,cB_source);


        // --------------------------------------------------------
        // Dirichlet B reservoir boundary
        // --------------------------------------------------------
        C[idCAbove] =cB_source;
        ++sourceCellCount;
    }

    // ============================================================
    // OPTIONAL DIAGNOSTICS
    // ============================================================

    static unsigned long callCount = 0;
    ++callCount;
    if (callCount % 100 == 0)
    {
        std::cout
            << "\n----- Li source / sink diagnostic -----"

            << "\nIteration calls        = "
            << callCount

            << "\nLi source cells        = "
            << sourceCellCount

            << "\nCondensing cells       = "
            << condensationCells

            << "\nMax source T           = "
            << maxSourceTemperature
            << " K"

            << "\nMax source cLi         = "
            << maxSourceConcentration
            << " mol/m^3"

            << "\nMax source delta cLi   = "
            << maxSourceDelta
            << " mol/m^3"

            << "\nRemoved concentration  = "
            << totalRemovedConcentration
            << " mol/m^3"

            << "\n---------------------------------------"
            << std::endl;
    }
}

//double sampleVaporPressure(double T, const double A, const double B, const double C)
//{
//    double log10P_bar =A - B / (T + C);
//
//    double P_bar =std::pow(10.0, log10P_bar);
//
//    return P_bar * 1.0e5;   // Pa
//
//}

double sampleVaporPressure(double T,double A,double B,double C,double D)
{
    const double log10P =A- B / T+ C * T+ D * std::log10(T);
    return std::pow(10.0, log10P)*torrToPa;
}


double calculateTotalNAr(
    const asl::SPDataWithGhostNodesACLData& nAr,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap)
{
    const auto& internal = nAr->getInternalBlock();
    const auto& blockN   = nAr->getBlock();
    const auto& blockM   = gasVolumeMap->getBlock();

    const auto size = internal.getSize();

    const int gn = nAr->getGhostBorder();
    const int gm = gasVolumeMap->getGhostBorder();

    auto nMapped =
        acl::map<double>(nAr->getContainer()[0]);

    auto mMapped =
        acl::map<double>(gasVolumeMap->getContainer()[0]);

    const double* N = nMapped.get();
    const double* M = mMapped.get();

    // dx is mm -> convert to metres
    const double dx_m = internal.dx * 1e-3;
    const double dV = dx_m * dx_m * dx_m;

    double totalNAr = 0.0;

    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idN = blockN.c2i(
            asl::makeAVec(i + gn, j + gn, k + gn));

        const int idM = blockM.c2i(
            asl::makeAVec(i + gm, j + gm, k + gm));

        if (M[idM] > 0.5)
            totalNAr += N[idN] * dV;
    }

    return totalNAr;
}


void updateNArFromXAr(
    const asl::SPDataWithGhostNodesACLData& nAr,
    const asl::SPDataWithGhostNodesACLData& xAr,
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double Peq_Pa)
{
    constexpr double kB =
        1.380649e-23;

    const auto& internal =
        nAr->getInternalBlock();

    const auto& bN  = nAr->getBlock();
    const auto& bX  = xAr->getBlock();
    const auto& bT  = temperature->getBlock();
    const auto& bM  = gasVolumeMap->getBlock();

    const auto size =
        internal.getSize();

    const int gn =
        nAr->getGhostBorder();

    const int gx =
        xAr->getGhostBorder();

    const int gt =
        temperature->getGhostBorder();

    const int gm =
        gasVolumeMap->getGhostBorder();


    auto nMap =
        acl::map<double>(
            nAr->getContainer()[0]
        );

    auto xMap =
        acl::map<double>(
            xAr->getContainer()[0]
        );

    auto tMap =
        acl::map<double>(
            temperature->getContainer()[0]
        );

    auto mMap =
        acl::map<double>(
            gasVolumeMap->getContainer()[0]
        );


    double* N =
        nMap.get();

    const double* X =
        xMap.get();

    const double* T =
        tMap.get();

    const double* M =
        mMap.get();


    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idN =
            bN.c2i(
                asl::makeAVec(
                    i + gn,
                    j + gn,
                    k + gn
                )
            );

        const int idX =
            bX.c2i(
                asl::makeAVec(
                    i + gx,
                    j + gx,
                    k + gx
                )
            );

        const int idT =
            bT.c2i(
                asl::makeAVec(
                    i + gt,
                    j + gt,
                    k + gt
                )
            );

        const int idM =
            bM.c2i(
                asl::makeAVec(
                    i + gm,
                    j + gm,
                    k + gm
                )
            );


        if (M[idM] > 0.5)
        {
            N[idN] =
                X[idX] *
                Peq_Pa /
                (kB * T[idT]);
        }
        else
        {
            N[idN] = 0.0;
        }
    }
}

double calculatePeq(
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double NAr)
{
    const auto& internalBlock = temperature->getInternalBlock();
    const auto& tempBlock     = temperature->getBlock();
    const auto& maskBlock     = gasVolumeMap->getBlock();

    const auto size = internalBlock.getSize();

    const int gt = temperature->getGhostBorder();
    const int gm = gasVolumeMap->getGhostBorder();

    auto tempMapped =acl::map<double>(temperature->getContainer()[0]);
    auto maskMapped =acl::map<double>(gasVolumeMap->getContainer()[0]);

    const double* T = tempMapped.get();
    const double* M = maskMapped.get();

    double integral = 0.0;

    // block.dx is in mm in our model
    const double dx_m = internalBlock.dx * 1e-3;
    const double dV   = dx_m * dx_m * dx_m;

    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idT = tempBlock.c2i(asl::makeAVec(i + gt, j + gt, k + gt));
        const int idM = maskBlock.c2i(asl::makeAVec(i + gm, j + gm, k + gm));
        if (M[idM] > 0.5) integral += dV / T[idT];
    }

    return NAr * kB / integral;      // Pa
}


asl::SPDataWithGhostNodesACLData buildGasVolumeMap(const asl::SPDataWithGhostNodesACLData& surface)
{
    // Same physical grid as the imported STL
    asl::Block block(surface->getInternalBlock());

    auto gasVolumeMap =
        asl::generateDataContainerACL_SP<double>(
            block,
            1,
            1u
        );

    const auto& surfBlock = surface->getBlock();
    const auto& gasBlock  = gasVolumeMap->getBlock();

    const int gs = surface->getGhostBorder();
    const int gm = gasVolumeMap->getGhostBorder();

    const auto size = block.getSize();

    const int Nz = size[0];
    const int Ny = size[1];
    const int Nx = size[2];

    // read STL-derived field
    auto surfMapped =
        acl::map<float>(surface->getContainer()[0]);

    // write 0/1 gas mask
    auto gasMapped =
        acl::map<double>(gasVolumeMap->getContainer()[0]);

    const float* phi = surfMapped.get();
    double* M        = gasMapped.get();


    // -------------------------------
    // Index helpers
    // -------------------------------

    auto logicalID = [&](int i, int j, int k)
    {
        return block.c2i(asl::makeAVec(i,j,k));
    };

    auto surfaceID = [&](int i, int j, int k)
    {
        return surfBlock.c2i(
            asl::makeAVec(i + gs,
                          j + gs,
                          k + gs)
        );
    };

    auto gasID = [&](int i, int j, int k)
    {
        return gasBlock.c2i(
            asl::makeAVec(i + gm,
                          j + gm,
                          k + gm)
        );
    };


    // phi <= 0 : voxel belonging to STL wall shell
    auto isWall = [&](int i, int j, int k)
    {
        return phi[surfaceID(i,j,k)] <= 0.0f;
    };


    // -------------------------------
    // Outside flags
    // -------------------------------

    std::vector<unsigned char>
        outside(Nz * Ny * Nx, 0);

    std::queue<std::array<int,3>> q;


    auto pushOutside =
        [&](int i, int j, int k)
    {
        const int lid = logicalID(i,j,k);

        if (!outside[lid] &&
            !isWall(i,j,k))
        {
            outside[lid] = 1;
            q.push({i,j,k});
        }
    };


    // -------------------------------
    // Seed computational-box faces
    // -------------------------------

    for (int i=0; i<Nz; ++i)
    for (int j=0; j<Ny; ++j)
    {
        pushOutside(i,j,0);
        pushOutside(i,j,Nx-1);
    }

    for (int i=0; i<Nz; ++i)
    for (int k=0; k<Nx; ++k)
    {
        pushOutside(i,0,k);
        pushOutside(i,Ny-1,k);
    }

    for (int j=0; j<Ny; ++j)
    for (int k=0; k<Nx; ++k)
    {
        pushOutside(0,j,k);
        pushOutside(Nz-1,j,k);
    }


    // -------------------------------
    // Six-neighbour flood fill
    // -------------------------------

    const int neighbour[6][3] =
    {
        { 1, 0, 0},
        {-1, 0, 0},
        { 0, 1, 0},
        { 0,-1, 0},
        { 0, 0, 1},
        { 0, 0,-1}
    };

    while (!q.empty())
    {
        auto p = q.front();
        q.pop();

        for (const auto& d : neighbour)
        {
            const int i = p[0] + d[0];
            const int j = p[1] + d[1];
            const int k = p[2] + d[2];

            if (i < 0 || i >= Nz ||
                j < 0 || j >= Ny ||
                k < 0 || k >= Nx)
                continue;

            pushOutside(i,j,k);
        }
    }


    // -------------------------------
    // Construct M_ijk
    // -------------------------------

    for (int i=0; i<Nz; ++i)
    for (int j=0; j<Ny; ++j)
    for (int k=0; k<Nx; ++k)
    {
        const int lid = logicalID(i,j,k);
        const int id  = gasID(i,j,k);

        if (!outside[lid] &&
            !isWall(i,j,k))
        {
            M[id] = 1.0;       // enclosed gas
        }
        else
        {
            M[id] = 0.0;       // outside / wall
        }
    }

    return gasVolumeMap;
}


void updatePeqField(
    const asl::SPDataWithGhostNodesACLData& PeqField,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double Peq)
{
    const auto& blockP = PeqField->getBlock();
    const auto& blockM = gasVolumeMap->getBlock();
    const auto& internal = PeqField->getInternalBlock();

    auto size = internal.getSize();

    int gp = PeqField->getGhostBorder();
    int gm = gasVolumeMap->getGhostBorder();

    auto pMapped =acl::map<double>(PeqField->getContainer()[0]);
    auto mMapped =acl::map<double>(gasVolumeMap->getContainer()[0]);

    double* P = pMapped.get();
    const double* M = mMapped.get();

    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        int idP = blockP.c2i(asl::makeAVec(i+gp, j+gp, k+gp));
        int idM = blockM.c2i(asl::makeAVec(i+gm, j+gm, k+gm));
        P[idP] = (M[idM] > 0.5) ? Peq : 0.0;
    }
}


double calculateGasVolume(
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap)
{
    const auto& internal = gasVolumeMap->getInternalBlock();
    const auto& blockM   = gasVolumeMap->getBlock();

    const auto size = internal.getSize();
    const int gm = gasVolumeMap->getGhostBorder();

    auto mMapped =acl::map<double>(gasVolumeMap->getContainer()[0]);

    const double* M = mMapped.get();

    // ASL geometry is in mm
    const double dx_m = internal.dx * 1.0e-3;
    const double dV   = dx_m * dx_m * dx_m;

    double volume = 0.0;

    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idM =blockM.c2i(asl::makeAVec(i + gm,j + gm,k + gm));
        if (M[idM] > 0.5)
            volume += dV;
    }

    return volume;   // m^3
}


// ============================================================
// BUILD SOURCE / SINK CACHE
//
// Called ONCE per temperature case AFTER T has converged.
//
// The geometry and temperature are fixed during the SM
// evolution, therefore:
//
//   * wall-adjacent gas cells are found once
//   * source cells are found once
//   * Psat(T) is calculated once
//   * cSat(T) is calculated once
//
// During SM evolution we therefore only touch the relevant
// cached cB cells.
// ============================================================

SourceSinkCache buildSourceSinkCache(
    const asl::SPDataWithGhostNodesACLData& cB,
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& SourceMap,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap)
{
    SourceSinkCache cache;

    // ------------------------------------------------------------
    // Internal computational region
    // ------------------------------------------------------------
    const auto& internal = cB->getInternalBlock();
    const auto size = internal.getSize();

    const int Nz = size[0];
    const int Ny = size[1];
    const int Nx = size[2];


    // ------------------------------------------------------------
    // Coordinate convention in this program:
    //
    //      component 0 -> z -> i
    //      component 1 -> y -> j
    //      component 2 -> x -> k
    //
    // The physical sample lies flat in the x-y plane.
    //
    // Therefore:
    //
    //      "above the sample" = +z = i + 1
    // ------------------------------------------------------------


    // ------------------------------------------------------------
    // ASL blocks
    // ------------------------------------------------------------
    const auto& bC = cB->getBlock();
    const auto& bT = temperature->getBlock();
    const auto& bS = SourceMap->getBlock();
    const auto& bG = gasVolumeMap->getBlock();


    // ------------------------------------------------------------
    // Ghost-border widths
    // ------------------------------------------------------------
    const int gc = cB->getGhostBorder();
    const int gt = temperature->getGhostBorder();
    const int gs = SourceMap->getGhostBorder();
    const int gg = gasVolumeMap->getGhostBorder();


    // ------------------------------------------------------------
    // Map FIXED fields to host memory ONCE
    // ------------------------------------------------------------
    auto temperatureMap =
        acl::map<double>(
            temperature->getContainer()[0]
        );

    auto sourceMap =
        acl::map<double>(
            SourceMap->getContainer()[0]
        );

    auto gasMap =
        acl::map<double>(
            gasVolumeMap->getContainer()[0]
        );


    const double* T =
        temperatureMap.get();

    const double* S =
        sourceMap.get();

    const double* G =
        gasMap.get();


    // ============================================================
    // HELPER:
    //
    // Determine whether logical voxel (i,j,k) is gas.
    //
    // Any cell outside the internal domain is treated as non-gas.
    // ============================================================

    auto isGas =
        [&](int i, int j, int k) -> bool
    {
        if (i < 0 || i >= Nz ||
            j < 0 || j >= Ny ||
            k < 0 || k >= Nx)
        {
            return false;
        }

        const int idG =
            bG.c2i(
                asl::makeAVec(
                    i + gg,
                    j + gg,
                    k + gg
                )
            );

        return G[idG] > 0.5;
    };


    // ============================================================
    // PART 1
    //
    // FIND ALL WALL-ADJACENT GAS CELLS
    //
    // These are the only locations where condensation is allowed.
    // ============================================================

    std::size_t totalGasCells = 0;


    for (int i = 0; i < Nz; ++i)
    for (int j = 0; j < Ny; ++j)
    for (int k = 0; k < Nx; ++k)
    {
        // --------------------------------------------------------
        // Current voxel must be gas
        // --------------------------------------------------------
        if (!isGas(i,j,k))
            continue;

        ++totalGasCells;


        // --------------------------------------------------------
        // Check six nearest neighbors
        //
        // If ANY neighbor is non-gas, this gas cell touches a wall.
        // --------------------------------------------------------

        const bool adjacentToWall =
            !isGas(i + 1, j,     k    ) ||
            !isGas(i - 1, j,     k    ) ||
            !isGas(i,     j + 1, k    ) ||
            !isGas(i,     j - 1, k    ) ||
            !isGas(i,     j,     k + 1) ||
            !isGas(i,     j,     k - 1);


        if (!adjacentToWall)
            continue;


        // --------------------------------------------------------
        // cB index
        // --------------------------------------------------------
        const int idC =
            bC.c2i(
                asl::makeAVec(
                    i + gc,
                    j + gc,
                    k + gc
                )
            );


        // --------------------------------------------------------
        // Temperature index
        // --------------------------------------------------------
        const int idT =
            bT.c2i(
                asl::makeAVec(
                    i + gt,
                    j + gt,
                    k + gt
                )
            );


        const double Tw =
            T[idT];


        if (Tw <= 0.0)
            continue;


        // --------------------------------------------------------
        // Local saturation vapor pressure
        //
        // sampleVaporPressure() returns Pa
        // --------------------------------------------------------
        const double PB_sat =sampleVaporPressure(Tw,A1,B1,C1, D1);

        // --------------------------------------------------------
        // Saturated molar concentration:
        //
        //      c_sat = P_sat / (R T)
        //
        // [mol/m^3]
        // --------------------------------------------------------
        const double cB_sat = PB_sat / (R * Tw);


        // --------------------------------------------------------
        // Store everything needed during SM evolution
        // --------------------------------------------------------
        cache.condensationCells.push_back(
            {
                idC,
                cB_sat
            }
        );
    }


    // ============================================================
    // PART 2
    //
    // SOURCE MAP ITSELF IS THE EMITTER
    //
    // Every voxel for which:
    //
    //      SourceMap(i,j,k) > 0.5
    //
    // is treated as an emitting reservoir voxel.
    //
    // There is NO:
    //
    //      +z search
    //      i + 1
    //      top-surface detection
    //      separate gas layer above the source
    //
    // The source concentration is imposed DIRECTLY on every
    // SourceMap voxel:
    //
    //      cB = Psat(T)/(R T)
    //
    // Therefore source thickness may be arbitrary.
    // ============================================================

    for (int i = 0; i < Nz; ++i)
    for (int j = 0; j < Ny; ++j)
    for (int k = 0; k < Nx; ++k)
    {
        // --------------------------------------------------------
        // SourceMap index
        // --------------------------------------------------------
        const int idS =bS.c2i(asl::makeAVec(i + gs,j + gs,k + gs));

        // --------------------------------------------------------
        // Not part of emitter
        // --------------------------------------------------------
        if (S[idS] <= 0.5)
            continue;

        // --------------------------------------------------------
        // Current voxel must belong to gas computational domain
        //
        // Since your buildSourceMap() currently creates source
        // cells only where gasVolumeMap > 0.5, this should normally
        // always pass.
        // --------------------------------------------------------
        const int idG =bG.c2i(asl::makeAVec(i + gg,j + gg,k + gg));

        if (G[idG] <= 0.5)
            continue;

        // --------------------------------------------------------
        // SAME voxel's B concentration index
        // --------------------------------------------------------
        const int idC = bC.c2i(asl::makeAVec(i + gc,j + gc,k + gc));

        // --------------------------------------------------------
        // SAME voxel's temperature index
        // --------------------------------------------------------
        const int idT =bT.c2i(asl::makeAVec(i + gt,j + gt,k + gt));
        const double Ts =T[idT];

        if (Ts <= 0.0)
            continue;


        // --------------------------------------------------------
        // Saturation vapor pressure at local source temperature
        // --------------------------------------------------------
        const double PB_sat =sampleVaporPressure(Ts,A1,B1,C1, D1);


        // --------------------------------------------------------
        // Saturated B concentration
        //
        //      cB_source = Psat(T)/(R T)
        //
        // --------------------------------------------------------
        const double cB_source =PB_sat /(R * Ts);

        // --------------------------------------------------------
        // Cache THIS SourceMap voxel itself as emitter
        // --------------------------------------------------------
        cache.sourceCells.push_back({idC,cB_source,Ts});
    }


    // ============================================================
    // CACHE DIAGNOSTICS
    // ============================================================

    const double wallFraction =
        (totalGasCells > 0)
        ?
        100.0 *
        static_cast<double>(
            cache.condensationCells.size()
        )
        /
        static_cast<double>(
            totalGasCells
        )
        :
        0.0;


    std::cout
        << "\n========================================"

        << "\nSOURCE / SINK CACHE CREATED"

        << "\nTotal gas cells         = "
        << totalGasCells

        << "\nWall-adjacent gas cells = "
        << cache.condensationCells.size()

        << "\nReservoir source cells  = "
        << cache.sourceCells.size()

        << "\nWall-cell fraction      = "
        << wallFraction
        << " %"

        << "\n========================================"

        << std::endl;


    return cache;
}



// ============================================================
// FAST SOURCE / SINK UPDATE
//
// Called EVERY Stefan-Maxwell iteration.
//
// Unlike the original function:
//
//   NO full 3-D scan
//   NO SourceMap mapping
//   NO gasVolumeMap mapping
//   NO temperature mapping
//   NO neighbor searching
//   NO sampleVaporPressure()
//   NO repeated c2i()
//
// Only the cached concentration indices are touched.
// ============================================================

void applySourceAndSinkFast(
    const asl::SPDataWithGhostNodesACLData& cB,
    const SourceSinkCache& cache)
{
    // ------------------------------------------------------------
    // Map ONLY changing concentration field
    // ------------------------------------------------------------
    auto cBMap =
        acl::map<double>(
            cB->getContainer()[0]
        );

    double* C =
        cBMap.get();


    // ============================================================
    // PART 1
    //
    // CONDENSATION SINK
    //
    // At wall-adjacent cells:
    //
    //      if cB > cSat(Twall)
    //
    // then:
    //
    //      cB -> cSat
    //
    // ============================================================

    double totalRemovedConcentration =
        0.0;

    unsigned int condensationCells =
        0;


    for (const auto& cell :
         cache.condensationCells)
    {
        const double current =
            C[cell.cIndex];


        if (current > cell.cSat)
        {
            totalRemovedConcentration +=
                current -
                cell.cSat;


            C[cell.cIndex] =
                cell.cSat;


            ++condensationCells;
        }
    }



    // ============================================================
    // PART 2
    //
    // RESERVOIR SOURCE
    //
    // Force gas immediately above sample to:
    //
    //      cB = Psat(Ts)/(R Ts)
    //
    // Applied AFTER condensation so the physical reservoir
    // boundary always wins.
    // ============================================================

    double maxSourceTemperature =
        0.0;

    double maxSourceConcentration =
        0.0;

    double maxSourceDelta =
        0.0;


    for (const auto& cell :
         cache.sourceCells)
    {
        const double oldConcentration =
            C[cell.cIndex];


        const double delta =
            cell.cSat -
            oldConcentration;


        maxSourceDelta =
            std::max(
                maxSourceDelta,
                std::abs(delta)
            );


        maxSourceTemperature =
            std::max(
                maxSourceTemperature,
                cell.temperature
            );


        maxSourceConcentration =
            std::max(
                maxSourceConcentration,
                cell.cSat
            );


        // --------------------------------------------------------
        // Dirichlet reservoir condition
        // --------------------------------------------------------
        C[cell.cIndex] =
            cell.cSat;
    }



    // ============================================================
    // DIAGNOSTICS
    // ============================================================

    static unsigned long callCount =
        0;

    ++callCount;


    if (callCount % 100 == 0)
    {
        std::cout
            << "\n----- B source / sink diagnostic -----"

            << "\nIteration calls        = "
            << callCount

            << "\nB source cells         = "
            << cache.sourceCells.size()

            << "\nCondensing cells       = "
            << condensationCells

            << "\nMax source T           = "
            << maxSourceTemperature
            << " K"

            << "\nMax source cB          = "
            << maxSourceConcentration
            << " mol/m^3"

            << "\nMax source delta cB    = "
            << maxSourceDelta
            << " mol/m^3"

            << "\nRemoved concentration  = "
            << totalRemovedConcentration
            << " mol/m^3"

            << "\n---------------------------------------"

            << std::endl;
    }
}

void copyField(
    const asl::SPDataWithGhostNodesACLData& dst,
    const asl::SPDataWithGhostNodesACLData& src)
{
    const auto& internal =src->getInternalBlock();
    const auto size =internal.getSize();
    const auto& bDst =dst->getBlock();
    const auto& bSrc =src->getBlock();
    const int gd =dst->getGhostBorder();
    const int gs =src->getGhostBorder();
    auto dstMap =acl::map<double>(dst->getContainer()[0]);
    auto srcMap = acl::map<double>(src->getContainer()[0]);
    double* D =dstMap.get();
    const double* S =srcMap.get();


    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idD =bDst.c2i(asl::makeAVec(i + gd,j + gd,k + gd));
        const int idS =bSrc.c2i(asl::makeAVec(i + gs,j + gs,k + gs));
        D[idD] = S[idS];
    }
}

TemperatureCubeResult checkTemperatureCube(
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double centerX,
    double centerY,
    double centerZ,
    double cubeLength_mm,
    double operatingT_K,
    double tolerance_K)
{
    // ============================================================
    // CHECK TEMPERATURE INSIDE A PHYSICAL CUBE
    //
    // User coordinate convention:
    //
    //      centerX [mm]
    //      centerY [mm]
    //      centerZ [mm]
    //
    // ASL indexing convention in this program:
    //
    //      i -> z
    //      j -> y
    //      k -> x
    //
    // Cube side length:
    //
    //      cubeLength_mm
    //
    // Convergence condition:
    //
    //      Tmin_cube >= operatingT_K - tolerance_K
    //
    // ============================================================

    TemperatureCubeResult result;

    if (cubeLength_mm <= 0.0)
    {
        throw std::runtime_error(
            "checkTemperatureCube(): cubeLength_mm must be > 0."
        );
    }


    // ------------------------------------------------------------
    // Internal block
    // ------------------------------------------------------------

    const auto& internal =temperature->getInternalBlock();
    const auto size =internal.getSize();
    const double dx = internal.dx;
    // ------------------------------------------------------------
    // Cube physical limits
    // ------------------------------------------------------------
    const double half =0.5 * cubeLength_mm;
    const double xmin =centerX - half;
    const double xmax =centerX + half;
    const double ymin =centerY - half;
    const double ymax =centerY + half;
    const double zmin =centerZ - half;
    const double zmax =centerZ + half;


    // ============================================================
    // Convert physical coordinates -> logical grid indices
    //
    // internal.position:
    //
    //      [0] -> z
    //      [1] -> y
    //      [2] -> x
    // ============================================================

    int i0 =static_cast<int>(std::ceil((zmin - internal.position[0]) / dx));
    int i1 =static_cast<int>(std::floor((zmax - internal.position[0]) / dx));
    int j0 =static_cast<int>(std::ceil((ymin - internal.position[1]) / dx));
    int j1 =static_cast<int>(std::floor((ymax - internal.position[1]) / dx));
    int k0 =static_cast<int>(std::ceil((xmin - internal.position[2]) / dx));
    int k1 =static_cast<int>(std::floor((xmax - internal.position[2]) / dx));
    // ------------------------------------------------------------
    // Make sure requested cube lies inside computational block
    // ------------------------------------------------------------

    if (i0 < 0 || i1 >= size[0] ||
        j0 < 0 || j1 >= size[1] ||
        k0 < 0 || k1 >= size[2])
    {
        throw std::runtime_error(
            "checkTemperatureCube(): requested cube extends "
            "outside computational domain."
        );
    }


    // ------------------------------------------------------------
    // ASL blocks
    // ------------------------------------------------------------

    const auto& bT = temperature->getBlock();
    const auto& bG =gasVolumeMap->getBlock();
    const int gt =temperature->getGhostBorder();
    const int gg =gasVolumeMap->getGhostBorder();

    // ------------------------------------------------------------
    // Map fields to host
    // ------------------------------------------------------------
    auto tMap =acl::map<double>(temperature->getContainer()[0]);
    auto gMap =acl::map<double>(gasVolumeMap->getContainer()[0]);
    const double* T =tMap.get();
    const double* G =gMap.get();

    // ============================================================
    // SEARCH CUBE
    // ============================================================

    double minimumT =std::numeric_limits<double>::infinity();
    double temperatureSum =0.0;
    std::size_t gasCells =0;

    for (int i = i0; i <= i1; ++i)
    for (int j = j0; j <= j1; ++j)
    for (int k = k0; k <= k1; ++k)
    {
        const int idG = bG.c2i(asl::makeAVec(i + gg,j + gg,k + gg ));
        // Ignore solid/outside cells
        if (G[idG] <= 0.5)
            continue;
        const int idT = bT.c2i(asl::makeAVec(i + gt,j + gt,k + gt ));
        const double localT =T[idT];
        minimumT =std::min(minimumT,localT);
        temperatureSum +=localT;
        ++gasCells;
    }


    // ------------------------------------------------------------
    // Cube must contain gas
    // ------------------------------------------------------------

    if (gasCells == 0)
    {
        throw std::runtime_error(
            "checkTemperatureCube(): cube contains no gas cells."
        );
    }


    // ------------------------------------------------------------
    // Average temperature
    // ------------------------------------------------------------
    const double averageT =temperatureSum /static_cast<double>(gasCells);

    // ============================================================
    // CONVERGENCE TEST
    //
    // The MINIMUM temperature is used as the gate.
    //
    // Therefore every gas voxel in the cube must be at least:
    //
    //      operatingT_K - tolerance_K
    // ============================================================
    const bool converged =minimumT >=(operatingT_K - tolerance_K);

        std::cout
        << " | cube center = ("
        << centerX << ", "
        << centerY << ", "
        << centerZ << ") mm"
        << " | cube length = "
        << cubeLength_mm
        << " mm\n"
        << " | Tmin = "
        << minimumT
        << " K"

        << " | Tavg = "
        << averageT
        << " K"

        << " | target = "
        << operatingT_K
        << " K\n"

        << " | converged = "
        << std::boolalpha
        << converged
        << std::endl;


    result.minT=minimumT;
    result.averageT=averageT;
    result.gasCells=gasCells;
    result.converged=converged;
    return result;
}


ConcentrationCubeResult checkConcentrationCube(
    const asl::SPDataWithGhostNodesACLData& cAr,
    const asl::SPDataWithGhostNodesACLData& cB,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,

    double centerX,
    double centerY,
    double centerZ,

    double cubeLength_mm,

    double relativeTolerance,
    double absoluteTolerance,

    ConcentrationConvergenceState& state)
{
    // ============================================================
    // CONCENTRATION CONVERGENCE INSIDE A PHYSICAL CUBE
    //
    // Coordinate convention:
    //
    //      i -> z
    //      j -> y
    //      k -> x
    //
    // Convergence:
    //
    // |c_new - c_old|
    //      <=
    // absTol + relTol * max(|c_new|, |c_old|)
    //
    // Must be satisfied for BOTH average cAr and average cB.
    // ============================================================

    if (cubeLength_mm <= 0.0)
    {
        throw std::runtime_error(
            "checkConcentrationCube(): cubeLength_mm must be > 0."
        );
    }


    // ------------------------------------------------------------
    // Internal computational block
    // ------------------------------------------------------------

    const auto& internal =
        cAr->getInternalBlock();

    const auto size =
        internal.getSize();

    const double dx =
        internal.dx;


    // ------------------------------------------------------------
    // Physical limits of cube
    // ------------------------------------------------------------

    const double half =
        0.5 * cubeLength_mm;


    const double xmin = centerX - half;
    const double xmax = centerX + half;

    const double ymin = centerY - half;
    const double ymax = centerY + half;

    const double zmin = centerZ - half;
    const double zmax = centerZ + half;


    // ============================================================
    // Convert physical coordinates to logical indices
    //
    // ASL:
    //
    //      [0] = z
    //      [1] = y
    //      [2] = x
    // ============================================================

    const int i0 =
        static_cast<int>(
            std::ceil(
                (zmin - internal.position[0]) / dx
            )
        );

    const int i1 =
        static_cast<int>(
            std::floor(
                (zmax - internal.position[0]) / dx
            )
        );


    const int j0 =
        static_cast<int>(
            std::ceil(
                (ymin - internal.position[1]) / dx
            )
        );

    const int j1 =
        static_cast<int>(
            std::floor(
                (ymax - internal.position[1]) / dx
            )
        );


    const int k0 =
        static_cast<int>(
            std::ceil(
                (xmin - internal.position[2]) / dx
            )
        );

    const int k1 =
        static_cast<int>(
            std::floor(
                (xmax - internal.position[2]) / dx
            )
        );


    // ------------------------------------------------------------
    // Check cube lies inside domain
    // ------------------------------------------------------------

    if (i0 < 0 || i1 >= size[0] ||
        j0 < 0 || j1 >= size[1] ||
        k0 < 0 || k1 >= size[2])
    {
        throw std::runtime_error(
            "checkConcentrationCube(): requested cube extends "
            "outside computational domain."
        );
    }


    // ------------------------------------------------------------
    // Blocks
    // ------------------------------------------------------------

    const auto& bAr =
        cAr->getBlock();

    const auto& bB =
        cB->getBlock();

    const auto& bG =
        gasVolumeMap->getBlock();


    const int gAr =
        cAr->getGhostBorder();

    const int gB =
        cB->getGhostBorder();

    const int gG =
        gasVolumeMap->getGhostBorder();


    // ------------------------------------------------------------
    // Map fields
    // ------------------------------------------------------------

    auto arMap =
        acl::map<double>(
            cAr->getContainer()[0]
        );

    auto bMap =
        acl::map<double>(
            cB->getContainer()[0]
        );

    auto gasMap =
        acl::map<double>(
            gasVolumeMap->getContainer()[0]
        );


    const double* Ar =
        arMap.get();

    const double* B =
        bMap.get();

    const double* G =
        gasMap.get();


    // ============================================================
    // STATISTICS
    // ============================================================

    double minCAr =
        std::numeric_limits<double>::infinity();

    double maxCAr =
        -std::numeric_limits<double>::infinity();

    double minCB =
        std::numeric_limits<double>::infinity();

    double maxCB =
        -std::numeric_limits<double>::infinity();


    double sumCAr = 0.0;
    double sumCB = 0.0;

    std::size_t gasCells = 0;


    // ============================================================
    // SEARCH CUBE
    // ============================================================

    for (int i = i0; i <= i1; ++i)
    for (int j = j0; j <= j1; ++j)
    for (int k = k0; k <= k1; ++k)
    {
        const int idG =
            bG.c2i(
                asl::makeAVec(
                    i + gG,
                    j + gG,
                    k + gG
                )
            );


        // Only physical gas cells
        if (G[idG] <= 0.5)
            continue;


        const int idAr =
            bAr.c2i(
                asl::makeAVec(
                    i + gAr,
                    j + gAr,
                    k + gAr
                )
            );


        const int idB =
            bB.c2i(
                asl::makeAVec(
                    i + gB,
                    j + gB,
                    k + gB
                )
            );


        const double localCAr =
            Ar[idAr];

        const double localCB =
            B[idB];


        // --------------------------------------------------------
        // Ar statistics
        // --------------------------------------------------------

        minCAr =
            std::min(
                minCAr,
                localCAr
            );

        maxCAr =
            std::max(
                maxCAr,
                localCAr
            );

        sumCAr += localCAr;


        // --------------------------------------------------------
        // B statistics
        // --------------------------------------------------------

        minCB =
            std::min(
                minCB,
                localCB
            );

        maxCB =
            std::max(
                maxCB,
                localCB
            );

        sumCB += localCB;


        ++gasCells;
    }


    if (gasCells == 0)
    {
        throw std::runtime_error(
            "checkConcentrationCube(): cube contains no gas cells."
        );
    }


    // ============================================================
    // AVERAGES
    // ============================================================

    const double averageCAr =
        sumCAr /
        static_cast<double>(gasCells);


    const double averageCB =
        sumCB /
        static_cast<double>(gasCells);


    ConcentrationCubeResult result;


    result.minCAr = minCAr;
    result.maxCAr = maxCAr;
    result.averageCAr = averageCAr;

    result.minCB = minCB;
    result.maxCB = maxCB;
    result.averageCB = averageCB;

    result.gasCells = gasCells;


    // ============================================================
    // FIRST CALL
    //
    // Nothing exists to compare against yet.
    // Store current values as reference.
    // ============================================================

    if (!state.initialized)
    {
        state.previousAverageCAr =
            averageCAr;

        state.previousAverageCB =
            averageCB;

        state.initialized = true;

        result.converged = false;

        return result;
    }


    // ============================================================
    // CHANGE SINCE PREVIOUS CHECK
    // ============================================================

    result.deltaCAr =
        std::abs(
            averageCAr -
            state.previousAverageCAr
        );


    result.deltaCB =
        std::abs(
            averageCB -
            state.previousAverageCB
        );


    // ------------------------------------------------------------
    // Denominators for reporting relative change
    // ------------------------------------------------------------

    const double scaleAr =
        std::max(
            std::abs(averageCAr),
            std::abs(state.previousAverageCAr)
        );


    const double scaleB =
        std::max(
            std::abs(averageCB),
            std::abs(state.previousAverageCB)
        );


    if (scaleAr > 0.0)
    {
        result.relativeChangeCAr =
            result.deltaCAr /
            scaleAr;
    }


    if (scaleB > 0.0)
    {
        result.relativeChangeCB =
            result.deltaCB /
            scaleB;
    }


    // ============================================================
    // CONVERGENCE CRITERIA
    // ============================================================

    const double allowedAr =
        absoluteTolerance +
        relativeTolerance * scaleAr;


    const double allowedB =
        absoluteTolerance +
        relativeTolerance * scaleB;


    const bool arConverged =
        result.deltaCAr <= allowedAr;


    const bool bConverged =
        result.deltaCB <= allowedB;


    result.converged =
        arConverged &&
        bConverged;


    // ============================================================
    // UPDATE REFERENCE FOR NEXT CHECK
    // ============================================================

    state.previousAverageCAr =
        averageCAr;

    state.previousAverageCB =
        averageCB;


        std::cout
        << "\nCube <cAr>    = "
        << result.averageCAr
        << " mol/m^3"

        << "\nCube <cB>     = "
        << result.averageCB
        << " mol/m^3"

        << "\nDelta <cAr>   = "
        << result.deltaCAr 

        << "\nDelta <cB>    = "
        << result.deltaCB

        << "\nRel change Ar = "
        << result.relativeChangeCAr

        << "\nRel change B  = "
        << result.relativeChangeCB

        << "\nConverged     = "
        << std::boolalpha
        << result.converged

        << "\n------------------------------"
        << std::endl;


    if (result.converged)
    {
        std::cout
            << "\n========================================"
            << "\nSTEFAN-MAXWELL CONCENTRATIONS CONVERGED"
            << "\n========================================"
            << std::endl;
    }

    return result;
}
