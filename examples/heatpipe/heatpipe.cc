#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include <queue>
#include <array>
#include <algorithm>


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
constexpr double A1 = 4.98831; //cus C is used elsewhere.
constexpr double B1 = 7918.984;
constexpr double C1 = -9.52;

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
double sampleVaporPressure(double T, const double A, const double B, const double C);
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
    const asl::SPDataWithGhostNodesACLData& cAr,
    const asl::SPDataWithGhostNodesACLData& cB,
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap);

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

    auto coldZPlus  =asl::generateDFCylinder(zr+5.0,asl::makeAVec(zcold+5, 0.0, 0.0), asl::makeAVec(PosZ, 0.0, 0.0));   
    //auto coldZMinus =asl::generateDFCylinder(zr+1,asl::makeAVec(zcold+5, 0.0, 0.0),asl::makeAVec(-PosZ, 0.0, 0.0));
    auto coldXPlus  =asl::generateDFCylinder(xr+5.0,asl::makeAVec(0.0, 0.0, xcold+5), asl::makeAVec(0.0, 0.0, PosX));
    auto coldXMinus =asl::generateDFCylinder(xr+5.0,asl::makeAVec(0.0, 0.0, xcold+5),asl::makeAVec(0.0, 0.0, -PosX));
    auto coldYPlus  =asl::generateDFCylinder(yr+5.0,asl::makeAVec(0.0, ycold+5, 0.0), asl::makeAVec(0.0,PosY, 0.0));    
    auto coldYMinus =asl::generateDFCylinder(yr+5.0,asl::makeAVec(0.0, ycold+5, 0.0),asl::makeAVec(0.0,-PosY, 0.0));

    //Turn stl discrete data into continuous distance function.
    auto cavityDF =std::make_shared<asl::DataInterpolation>(heatpipe_cavity); 
    
    //auto coldRegion =zPlusPlane| zMinusPlane| xPlusPlane| xMinusPlane;
    auto coldRegion =coldZPlus | coldXMinus|coldXPlus | coldYMinus|coldYPlus;
    auto coldWall =  cavityDF & coldRegion;
    auto hotWall  =  cavityDF & (-coldRegion);

    auto coldWallMap =asl::generateDataContainerACL_SP<FlT>(block,1,1u);
    auto hotWallMap = asl::generateDataContainerACL_SP<FlT>(block,1,1u);

    //initialize boundary maps
    std::cout<<"Initializing different heatpipe walls+windows with correct boundary condition temperatures"<<std::endl;
    asl::initData(coldWallMap,asl::normalize(coldWall, dx));
    asl::initData(hotWallMap,asl::normalize(hotWall, dx));
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

    // ============================================================
    // QUASI-STATIC TEMPERATURE CASES
    // ============================================================
    // Number of numerical iterations.
    // These DO NOT represent physical seconds.
    // We are simply advancing the fields toward steady state.
    const unsigned int thermalSteps = 2000;
    const unsigned int smSteps      = 2000;

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
            nm->execute();
            executeAll(thermalbc);
            if (!(iteration % 100))
            {
                std::cout<< "  thermal iteration "<< iteration<< " / "<< thermalSteps<< std::endl;
                //writer.write();
            }
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
        updatePeqField(PeqField,gasVolumeMap,Peq_Torr);

        initializeCAr(cAr, temperature, gasVolumeMap, Peq_Pa);
        updateNumberDensity(nAr,cAr,gasVolumeMap);
        updatePressureFields(PArField,PBField,PtotalField,cAr,cB,temperature,gasVolumeMap);
   
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

        //Ar-B TRANSPORT USING STEFAN-MAXWELL'S DIFFUSION 
        std::cout << "Evolving Ar-B diffusion...\n";
        for (unsigned int iteration = 1;iteration <= smSteps;++iteration)
        {
            smTransport->execute(); // Coupled Li-Ar Stefan-Maxwell update
            executeAll(smBC);       // No species transport through steel walls

            // B reservoir restores saturated composition
            applySourceAndSinkFast(cB,sourceSinkCache);
            //applySourceAndSink(cB,temperature,SourceMap,gasVolumeMap);
        
            if (!(iteration % 1000))
            {
                updateNumberDensity(nB,cB,gasVolumeMap);
                updateNumberDensity(nAr,cAr,gasVolumeMap);
                updateMoleFractions(xB,xAr,cB,cAr,gasVolumeMap);
                updatePressureFields(PArField,PBField,PtotalField,cAr,cB,temperature,gasVolumeMap);
                writer.write();
                std::cout<< "  STEFAN-MAXWELL iteration "<< iteration<< " / "<< smSteps<< std::endl;
            }
        }

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
    const asl::SPDataWithGhostNodesACLData& cAr,
    const asl::SPDataWithGhostNodesACLData& cB,
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap)
{
    const auto& internal = cAr->getInternalBlock();
    const auto size = internal.getSize();

    const auto& bPA = PAr->getBlock();
    const auto& bPB = PB->getBlock();
    const auto& bPT = Ptotal->getBlock();
    const auto& bCA = cAr->getBlock();
    const auto& bCB = cB->getBlock();
    const auto& bT  = temperature->getBlock();
    const auto& bG  = gasVolumeMap->getBlock();

    const int gPA = PAr->getGhostBorder();
    const int gPB = PB->getGhostBorder();
    const int gPT = Ptotal->getGhostBorder();
    const int gCA = cAr->getGhostBorder();
    const int gCB = cB->getGhostBorder();
    const int gT  = temperature->getGhostBorder();
    const int gG  = gasVolumeMap->getGhostBorder();

    auto paMap = acl::map<double>(PAr->getContainer()[0]);
    auto pbMap = acl::map<double>(PB->getContainer()[0]);
    auto ptMap = acl::map<double>(Ptotal->getContainer()[0]);
    auto caMap = acl::map<double>(cAr->getContainer()[0]);
    auto cbMap = acl::map<double>(cB->getContainer()[0]);
    auto tMap  = acl::map<double>(temperature->getContainer()[0]);
    auto gMap  = acl::map<double>(gasVolumeMap->getContainer()[0]);

    double* PA = paMap.get();
    double* PBv = pbMap.get();
    double* PT = ptMap.get();

    const double* CA = caMap.get();
    const double* CB = cbMap.get();
    const double* T = tMap.get();
    const double* G = gMap.get();

    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idPA = bPA.c2i(asl::makeAVec(i+gPA,j+gPA,k+gPA));
        const int idPB = bPB.c2i(asl::makeAVec(i+gPB,j+gPB,k+gPB));
        const int idPT = bPT.c2i(asl::makeAVec(i+gPT,j+gPT,k+gPT));
        const int idCA = bCA.c2i(asl::makeAVec(i+gCA,j+gCA,k+gCA));
        const int idCB = bCB.c2i(asl::makeAVec(i+gCB,j+gCB,k+gCB));
        const int idT  = bT.c2i(asl::makeAVec(i+gT,j+gT,k+gT));
        const int idG  = bG.c2i(asl::makeAVec(i+gG,j+gG,k+gG));

        if (G[idG] > 0.5)
        {
            const double PAr_Pa =std::max(0.0, CA[idCA]) * R * T[idT];
            const double PB_Pa =std::max(0.0, CB[idCB]) * R * T[idT];

            PA[idPA]  = PAr_Pa / torrToPa;
            PBv[idPB] = PB_Pa  / torrToPa;
            PT[idPT]  = (PAr_Pa + PB_Pa) / torrToPa;
        }
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
        const double PB_sat =sampleVaporPressure(Tw,A1,B1,C1);     // Pa

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
        const double PB_sat =sampleVaporPressure(Ts, A1, B1, C1);

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

double sampleVaporPressure(double T, const double A, const double B, const double C)
{
    double log10P_bar =A - B / (T + C);

    double P_bar =std::pow(10.0, log10P_bar);

    return P_bar * 1.0e5;   // Pa

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


SourceSinkCache buildSourceSinkCache(
    const asl::SPDataWithGhostNodesACLData& cB,
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& SourceMap,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap)
{
    SourceSinkCache cache;

    // ------------------------------------------------------------
    // Internal region
    // ------------------------------------------------------------
    const auto& internal = cB->getInternalBlock();
    const auto size = internal.getSize();

    const int Nz = size[0];
    const int Ny = size[1];
    const int Nx = size[2];

    // ------------------------------------------------------------
    // ASL blocks
    // ------------------------------------------------------------
    const auto& bC = cB->getBlock();
    const auto& bT = temperature->getBlock();
    const auto& bS = SourceMap->getBlock();
    const auto& bG = gasVolumeMap->getBlock();

    // ------------------------------------------------------------
    // Ghost widths
    // ------------------------------------------------------------
    const int gc = cB->getGhostBorder();
    const int gt = temperature->getGhostBorder();
    const int gs = SourceMap->getGhostBorder();
    const int gg = gasVolumeMap->getGhostBorder();

    // ------------------------------------------------------------
    // Map the FIXED fields to host memory ONCE
    // ------------------------------------------------------------
    auto temperatureMap =
        acl::map<double>(temperature->getContainer()[0]);

    auto sourceMap =
        acl::map<double>(SourceMap->getContainer()[0]);

    auto gasMap =
        acl::map<double>(gasVolumeMap->getContainer()[0]);

    const double* T = temperatureMap.get();
    const double* S = sourceMap.get();
    const double* G = gasMap.get();

    // ------------------------------------------------------------
    // Helper: is logical cell (i,j,k) a gas cell?
    //
    // Outside computational domain is treated as non-gas.
    // This also avoids unsafe i-1, j-1, etc. indexing.
    // ------------------------------------------------------------
    auto isGas = [&](int i, int j, int k) -> bool
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
    // ============================================================

    for (int i = 0; i < Nz; ++i)
    for (int j = 0; j < Ny; ++j)
    for (int k = 0; k < Nx; ++k)
    {
        // Current cell must be gas
        if (!isGas(i,j,k))
            continue;

        // --------------------------------------------------------
        // Is this gas cell touching a wall/non-gas cell?
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

        const double Tw = T[idT];

        if (Tw <= 0.0)
            continue;

        // --------------------------------------------------------
        // Saturated B pressure
        // --------------------------------------------------------
        const double PB_sat =
            sampleVaporPressure(
                Tw,
                A1,
                B1,
                C1
            );

        // --------------------------------------------------------
        // Saturated molar concentration
        //
        //       c_sat = P_sat / (R T)
        // --------------------------------------------------------
        const double cB_sat =
            PB_sat / (R * Tw);

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
    // FIND GAS CELLS DIRECTLY ABOVE RESERVOIR
    //
    // Coordinate convention:
    //
    //      i -> z
    //      j -> y
    //      k -> x
    //
    // Therefore ABOVE means:
    //
    //      j + 1
    //
    // NOT i + 1.
    // ============================================================

    for (int i = 0; i < Nz; ++i)
    for (int j = 0; j < Ny; ++j)
    for (int k = 0; k < Nx; ++k)
    {
        const int idS =
            bS.c2i(
                asl::makeAVec(
                    i + gs,
                    j + gs,
                    k + gs
                )
            );

        // Current cell is not part of reservoir
        if (S[idS] <= 0.5)
            continue;

        // No interior cell above this one
        if (j + 1 >= Ny)
            continue;

        // --------------------------------------------------------
        // IMPORTANT FIX:
        //
        // +y = j + 1
        //
        // Your old code incorrectly used i + 1.
        // --------------------------------------------------------
        const int idSAbove =
            bS.c2i(
                asl::makeAVec(
                    i + gs,
                    j + 1 + gs,
                    k + gs
                )
            );

        // If another source voxel is above, current voxel
        // is not the top surface.
        if (S[idSAbove] > 0.5)
            continue;

        // --------------------------------------------------------
        // Gas immediately above source?
        // --------------------------------------------------------
        const int idGAbove =
            bG.c2i(
                asl::makeAVec(
                    i + gg,
                    j + 1 + gg,
                    k + gg
                )
            );

        if (G[idGAbove] <= 0.5)
            continue;

        // --------------------------------------------------------
        // cB and T indices of gas cell above reservoir
        // --------------------------------------------------------
        const int idCAbove =
            bC.c2i(
                asl::makeAVec(
                    i + gc,
                    j + 1 + gc,
                    k + gc
                )
            );

        const int idTAbove =
            bT.c2i(
                asl::makeAVec(
                    i + gt,
                    j + 1 + gt,
                    k + gt
                )
            );

        const double Ts = T[idTAbove];

        if (Ts <= 0.0)
            continue;

        // --------------------------------------------------------
        // Reservoir saturation pressure
        // --------------------------------------------------------
        const double PB_sat =
            sampleVaporPressure(
                Ts,
                A1,
                B1,
                C1
            );

        // --------------------------------------------------------
        // Reservoir concentration
        // --------------------------------------------------------
        const double cB_source =
            PB_sat / (R * Ts);

        cache.sourceCells.push_back(
            {
                idCAbove,
                cB_source,
                Ts
            }
        );
    }


    // ============================================================
    // DIAGNOSTIC
    // ============================================================

    std::cout
        << "\n========================================"
        << "\nSOURCE / SINK CACHE CREATED"
        << "\nWall-adjacent gas cells = "
        << cache.condensationCells.size()
        << "\nReservoir source cells  = "
        << cache.sourceCells.size()
        << "\n========================================"
        << std::endl;


    return cache;
}


// ============================================================
// FAST SOURCE / SINK UPDATE
//
// This is the ONLY function called inside the SM loop.
//
// No:
//
//   * full 3-D scan
//   * gasVolumeMap mapping
//   * SourceMap mapping
//   * temperature mapping
//   * neighbor searching
//   * c2i calculations
//   * vapor-pressure calculations
//
// Only cB is mapped.
// ============================================================

void applySourceAndSinkFast(
    const asl::SPDataWithGhostNodesACLData& cB,
    const SourceSinkCache& cache)
{
    // ------------------------------------------------------------
    // Map ONLY the changing B concentration field
    // ------------------------------------------------------------
    auto cBMap =
        acl::map<double>(
            cB->getContainer()[0]
        );

    double* C = cBMap.get();


    // ============================================================
    // PART 1
    // CONDENSATION
    //
    // cB -> min(cB, cSat)
    // ============================================================

    double totalRemovedConcentration = 0.0;
    unsigned int condensationCells = 0;

    for (const auto& cell : cache.condensationCells)
    {
        const double oldValue = C[cell.cIndex];

        if (oldValue > cell.cSat)
        {
            totalRemovedConcentration +=
                oldValue - cell.cSat;

            C[cell.cIndex] =
                cell.cSat;

            ++condensationCells;
        }
    }


    // ============================================================
    // PART 2
    // RESERVOIR SOURCE
    //
    // Source is applied AFTER sink so source boundary wins.
    // ============================================================

    double maxSourceTemperature = 0.0;
    double maxSourceConcentration = 0.0;
    double maxSourceDelta = 0.0;

    for (const auto& cell : cache.sourceCells)
    {
        const double delta =
            cell.cSat - C[cell.cIndex];

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

    static unsigned long callCount = 0;
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
