#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include <queue>
#include <array>


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



//declaraation of the temperature function
struct WallTemperatureParameters
{
    double Tmin = 19.0 + 273.15;
    double Tmax = 588.0 + 273.15;

    double zHot = 110.0;
    double wz   = 15.0;

    double xHot = 30.0;
    double wx   = 8.0;

    double Rmain = 15.0;
    double Rarm  = 10.0;
};
double TemperatureFunction(double x, double y, double z, const WallTemperatureParameters&);
asl::SPDataWithGhostNodesACLData buildGasVolumeMap(const asl::SPDataWithGhostNodesACLData& surface);
double calculatePeq(const asl::SPDataWithGhostNodesACLData& temperature,const asl::SPDataWithGhostNodesACLData& gasVolumeMap,double NAr);
void updatePeqField(const asl::SPDataWithGhostNodesACLData& PeqField,const asl::SPDataWithGhostNodesACLData& gasVolumeMap,double Peq);
void updateNAr(const asl::SPDataWithGhostNodesACLData& nAr,const asl::SPDataWithGhostNodesACLData& temperature,const asl::SPDataWithGhostNodesACLData& gasVolumeMap,double Peq_Pa);
double calculateTotalNAr(const asl::SPDataWithGhostNodesACLData& nAr,const asl::SPDataWithGhostNodesACLData& gasVolumeMap);

typedef float FlT;

int main()
{
    std::cout << "Starting STL import test..." << std::endl;

    const std::string stlFile = "C:/_dev/numerical_solvers/ASL-0.1.7/examples/heatpipe/heatpipe_fluid.stl";


    //setup the computational block
    double dx = 1.0; //step size
                                      //z,   y,x
    asl::AVec<int> size =asl::makeAVec(701, 81,151);
    asl::AVec<> origin =asl::makeAVec(-350.0,-40.0, -75.0);//-155.0);

    asl::Block initialBlock(size,dx,origin);

    std::cout << "Computational block created." << std::endl;

    //[min,max]  => x∈[−25,25] mm,y∈[−10,10] mm,z∈[−150,150] mm. for the heatpipe_fluid.stl
    //auto heatpipe_cavity =asl::readSurface(stlFile,initialBlock);
    auto heatpipe_cavity =asl::readSurface(stlFile,dx);


    if (!heatpipe_cavity)
    {
        std::cerr
            << "ERROR: Could not read STL."
            << std::endl;

        return 1;
    }

    std::cout<< "STL read successfully." << std::endl;
    asl::Block block(heatpipe_cavity->getInternalBlock());
    std::cout<< "Internal block created." << std::endl;
    std::cout<< "Grid size: "<< block.getSize()[0] << " x "<< block.getSize()[1] << " x "<< block.getSize()[2]<< std::endl;

    

    // ---------------------------------------------------------
    // Write ASL's STL-derived distance field
    // ---------------------------------------------------------

    std::cout << "Writing cavity geometry..." << std::endl;

    asl::WriterVTKXML writer("heatpipe_geometry");

    writer.addScalars("cavity",*heatpipe_cavity);


    std::cout<< "Geometry written successfully."<< std::endl;

    auto gasVolumeMap = buildGasVolumeMap(heatpipe_cavity);
    writer.addScalars("gasVolumeMap", *gasVolumeMap);
    //temperature scalar field on the surface of the heatpipe.
    auto temperature =asl::generateDataContainerACL_SP<double>
    (
        block,   //3d region
        1,      // one scalar component
        1u      // ghost/border layer
    );

    auto nAr =asl::generateDataContainerACL_SP<double>(
        block,
        1,
        1u
    );

    auto PeqField = asl::generateDataContainerACL_SP<double>(
        block,
        1,
        1u
    );


    //boundary conditions.
    std::vector<asl::SPNumMethod> bc;
    const double Tmax=388.0+273.15;
    const double Tmin = 19.1+273.15;
    const double P0_Torr = 5.0;


    const double torrToPa = 133.322368;
    const double P0_Pa =  P0_Torr*torrToPa;
    const double kB = 1.380649e-23;

    //r_main and r_arm and l_main, l_arm;
    const double r_main=15.0, r_arm = 10.0; //mm
    const double l_main = 150.0, l_arm=25.0;
    double V =  M_PI*r_main*r_main*l_main + 1.0/2.0*M_PI*r_arm*r_arm*l_arm; //- some overlap volumn. 
    const double V_gas=V*1e-9;//in m^3.
    std::cout<<"Calculated volume of the gas chamber in m^3 "<<V_gas<<std::endl;
    const double NAr = P0_Pa * V_gas / (kB * Tmin);
    std::cout<<"Calculated NAr "<<NAr<<std::endl;

    //for integration.
    const double dV = std::pow(dx*1e-3,3);


    




    //now the scalar function on the heatpipe surface. This is the temperature distribution on the heatpipe surface.
    auto x = asl::generatePFLinear(asl::makeAVec(0.0, 0.0, 1.0),0.0); //along x-axis
    auto z = asl::generatePFLinear(asl::makeAVec(1.0, 0.0, 0.0),0.0); //along z-axis


    const double Rmain = 18.0; //Long Cylinder inner diameter is 30mm
    const double LcoldZ = 30.0;
    const double PosZ   = 290.0;

    auto coldZPlus =asl::generateDFCylinder(Rmain,asl::makeAVec(LcoldZ, 0.0, 0.0), asl::makeAVec(PosZ+1, 0.0, 0.0));    // center: z = +145);
    auto coldZMinus =asl::generateDFCylinder(Rmain,asl::makeAVec(LcoldZ, 0.0, 0.0),asl::makeAVec(-PosZ, 0.0, 0.0));
    

    const double Rarm = 13.0;
    const double LcoldX = 25.0;
    const double PosX   = 40.0;

    auto coldXPlus =asl::generateDFCylinder(Rarm,asl::makeAVec(0.0, 0.0, LcoldX), asl::makeAVec(1.0, 0.0, PosX));
    auto coldXMinus =asl::generateDFCylinder(Rarm,asl::makeAVec(0.0, 0.0, LcoldX),asl::makeAVec(1.0, 0.0, -PosX));


    //Turn stl discrete data into continuous distance function.
    auto cavityDF =std::make_shared<asl::DataInterpolation>(heatpipe_cavity); 
    

    //auto coldRegion =zPlusPlane| zMinusPlane| xPlusPlane| xMinusPlane;
    auto coldRegion =coldZPlus|coldZMinus | coldXMinus|coldXPlus;
    auto coldWall =  cavityDF & coldRegion;
    auto hotWall  =  cavityDF & (-coldRegion);

    //
    auto coldWallMap =asl::generateDataContainerACL_SP<FlT>(block,1,1u);
    auto hotWallMap = asl::generateDataContainerACL_SP<FlT>(block,1,1u);

    //initialize boundary maps
    asl::initData(coldWallMap,asl::normalize(coldWall, dx));
    asl::initData(hotWallMap,asl::normalize(hotWall, dx));


    //imposing boundary conditins.
    bc.push_back(asl::generateBCConstantValue(temperature,Tmax,hotWallMap));
    bc.push_back(asl::generateBCConstantValue(temperature,Tmin,coldWallMap));
    bc.push_back(asl::generateBCConstantValue(temperature,Tmin,coldWallMap));
    asl::initAll(bc);


    //paraview debug
    writer.addScalars("coldWallMap", *coldWallMap);
    writer.addScalars("hotWallMap",  *hotWallMap);


    //initialize the temperature field.
    asl::initData(temperature,Tmin);

    //diffusuon solver
    auto templ = &asl::d3q15();
    double dt = 0.01;        // s
    double alpha = 10.0;     // mm^2/s, temporary test value
    double diffCoefNum =alpha * dt / (dx * dx);
    auto nm =generateFDAdvectionDiffusion(temperature,diffCoefNum,templ);

    nm->init();

    //execute bc once for initial state.
    executeAll(bc);


    writer.addScalars("temperature", *temperature);
    writer.addScalars("Peq_Torr", *PeqField);
    writer.addScalars("nAr", *nAr);
    for (unsigned int i = 1; i <= 100; ++i)
    {
        nm->execute();
        executeAll(bc);


        if (!(i % 10))
        {
            std::cout << "Step " << i << std::endl;
                    
            double Peq=calculatePeq(temperature, gasVolumeMap, NAr);
            double Peq_Torr =  Peq/torrToPa;
            updatePeqField(PeqField,gasVolumeMap,Peq_Torr);

            updateNAr(nAr,temperature,gasVolumeMap,Peq);
            //double NAr_check =calculateTotalNAr(nAr, gasVolumeMap);

            std::cout << "NAr initial = " << NAr
            //<< "\nNAr check   = " << NAr_check
            //<< "\nrelative error = "
            //<< std::abs(NAr_check - NAr) / NAr
            << std::endl;
          
            writer.write();
        }
    }

    
    //writer.write();
    std::cout
        << "=== Finished ==="
        << std::endl;

    return 0;
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

void updateNAr(
    const asl::SPDataWithGhostNodesACLData& nAr,
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double Peq_Pa)
{
    constexpr double kB = 1.380649e-23;

    const auto& internal = temperature->getInternalBlock();

    const auto& blockN = nAr->getBlock();
    const auto& blockT = temperature->getBlock();
    const auto& blockM = gasVolumeMap->getBlock();

    const auto size = internal.getSize();

    const int gn = nAr->getGhostBorder();
    const int gt = temperature->getGhostBorder();
    const int gm = gasVolumeMap->getGhostBorder();

    auto nMapped = acl::map<double>(nAr->getContainer()[0]);
    auto tMapped = acl::map<double>(temperature->getContainer()[0]);
    auto mMapped = acl::map<double>(gasVolumeMap->getContainer()[0]);

    double* N = nMapped.get();
    const double* T = tMapped.get();
    const double* M = mMapped.get();

    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        const int idN = blockN.c2i(
            asl::makeAVec(i + gn, j + gn, k + gn));

        const int idT = blockT.c2i(
            asl::makeAVec(i + gt, j + gt, k + gt));

        const int idM = blockM.c2i(
            asl::makeAVec(i + gm, j + gm, k + gm));

        if (M[idM] > 0.5)
            N[idN] = Peq_Pa / (kB * T[idT]);
        else
            N[idN] = 0.0;
    }
}

double calculatePeq(
    const asl::SPDataWithGhostNodesACLData& temperature,
    const asl::SPDataWithGhostNodesACLData& gasVolumeMap,
    double NAr)
{
    constexpr double kB = 1.380649e-23;   // J/K

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

    auto pMapped =
        acl::map<double>(PeqField->getContainer()[0]);

    auto mMapped =
        acl::map<double>(gasVolumeMap->getContainer()[0]);

    double* P = pMapped.get();
    const double* M = mMapped.get();

    for (int i = 0; i < size[0]; ++i)
    for (int j = 0; j < size[1]; ++j)
    for (int k = 0; k < size[2]; ++k)
    {
        int idP = blockP.c2i(
            asl::makeAVec(i+gp, j+gp, k+gp));

        int idM = blockM.c2i(
            asl::makeAVec(i+gm, j+gm, k+gm));

        P[idP] = (M[idM] > 0.5) ? Peq : 0.0;
    }
}

//scalar field functions
double TemperatureFunction(double x, double y, double z, const WallTemperatureParameters &p)
{
    // Main longitudinal tube, along z
    const double zHot = p.zHot;   // mm
    const double wz   = p.wz;    // mm

    // Two lateral arms, along +/-x
    const double xHot = p.xHot;    // mm
    const double wx   = p.wx;     // mm

    const double Rmain = p.Rmain;   // mm
    const double Rarm  = p.Rarm;   // mm

    // Smooth axial profile for main tube
    const double fz =0.5 *(std::tanh((z + zHot) / wz)-std::tanh((z - zHot) / wz));

    // Smooth symmetric profile for both lateral arms
    const double fx =0.5 *(1.0 -std::tanh((std::abs(x) - xHot) / wx));

    // Decide whether this location belongs primarily
    // to the lateral-arm region.
    const bool inArmRegion =(std::abs(z) <= Rarm) &&(std::abs(x) >= Rmain);

    const double f =inArmRegion ? fx : fz;

    return p.Tmin + (p.Tmax - p.Tmin) * f;
}