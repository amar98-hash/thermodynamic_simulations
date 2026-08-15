#include <iostream>
#include <iomanip>

#include <utilities/aslParametersManager.h>
#include <math/aslTemplates.h>

#include <aslGeomInc.h>
#include <aslDataInc.h>

#include <acl/aclGenerators.h>

#include <writers/aslVTKFormatWriters.h>

// Heat equation solver headers.
// We will actually use these in the NEXT step.
#include <num/aslFDAdvectionDiffusion.h>
#include <num/aslBasicBC.h>

//
#include <math/aslPositionFunction.h>

//fluid flows
#include <num/aslLBGK.h>
#include <num/aslLBGKBC.h>

#include <acl/DataTypes/aclMemBlock.h>

typedef float FlT;

using asl::AVec;
using asl::makeAVec;


// ================================================================
// MAIN
// ================================================================

int main(int argc, char* argv[])
{
    // ============================================================
    // 1. APPLICATION PARAMETERS
    // ============================================================

    asl::ApplicationParametersManager appParamsManager(
        "heatpipe_init",
        "0.3"
    );


    // ------------------------------------------------------------
    // Numerical grid
    // ------------------------------------------------------------

    asl::Parameter<FlT> dx(
        0.002f,
        "dx",
        "grid spacing",
        "m"
    );

    asl::Parameter<unsigned int> nx(
        200,
        "nx",
        "number of grid points along heat-pipe axis"
    );

    asl::Parameter<unsigned int> ny(
        40,
        "ny",
        "number of grid points in y"
    );

    asl::Parameter<unsigned int> nz(
        40,
        "nz",
        "number of grid points in z"
    );


    // ------------------------------------------------------------
    // Heat-pipe geometry
    // ------------------------------------------------------------

    asl::Parameter<FlT> pipeRadius(
        0.025f,
        "radius",
        "internal heat-pipe radius",
        "m"
    );


    // ------------------------------------------------------------
    // Temperature parameters
    // ------------------------------------------------------------

    asl::Parameter<FlT> Tcold(
        292.15f,
        "Tcold",
        "cold-window temperature",
        "K"
    );

    asl::Parameter<FlT> Thot(
        861.15f,
        "Thot",
        "hot-wall temperature",
        "K"
    );


    // ------------------------------------------------------------
    // Heat-equation numerical parameters
    // ------------------------------------------------------------

    asl::Parameter<FlT> dt(
        0.001f,
        "dt",
        "heat-equation time step",
        "s"
    );

    asl::Parameter<FlT> alpha(
        1.0e-4f,
        "alpha",
        "thermal diffusivity",
        "m^2/s"
    );


    asl::Parameter<FlT> nu(
    2.5e-5f,
    "nu",
    "kinematic viscosity for initial flow test",
    "m^2/s"
);

    // ------------------------------------------------------------
    // Intended thermal-zone geometry
    //
    // We are NOT imposing this profile on the gas anymore.
    //
    // These values will later define the wall BC:
    //
    // cold | transition | hot | transition | cold
    // ------------------------------------------------------------

    asl::Parameter<FlT> coldLength(
        0.05f,
        "coldLength",
        "length of cold region at each end",
        "m"
    );

    asl::Parameter<FlT> transitionLength(
        0.05f,
        "transitionLength",
        "length of each thermal transition",
        "m"
    );


    // ------------------------------------------------------------
    // Argon pressure
    // ------------------------------------------------------------

    asl::Parameter<FlT> pArTorr(
        5.0f,
        "PAr",
        "cold-fill argon pressure",
        "Torr"
    );


    // ------------------------------------------------------------
    // Parse command-line arguments
    // ------------------------------------------------------------

    appParamsManager.load(argc, argv);


    // ============================================================
    // 2. PHYSICAL CONSTANTS
    // ============================================================

    const double kB =
        1.380649e-23;          // J/K

    const double R =
        8.31446261815324;      // J/(mol K)

    const double torrToPa =
        133.32236842105263;    // Pa/Torr

    const double molarMassAr =
        39.948e-3;             // kg/mol


    // ============================================================
    // 3. NUMERICAL DIFFUSION COEFFICIENT
    //
    //          alpha * dt
    // alphaNum = ----------
    //             dx^2
    //
    // This is the nondimensional coefficient ASL will use later.
    // ============================================================

    const double alphaNum =
        static_cast<double>(alpha.v()) *
        static_cast<double>(dt.v())
        /
        (
            static_cast<double>(dx.v()) *
            static_cast<double>(dx.v())
        );


    const double nuNum =
    static_cast<double>(nu.v()) *
    static_cast<double>(dt.v()) /
    (
        static_cast<double>(dx.v()) *
        static_cast<double>(dx.v())
    );

    const double Lcold =static_cast<double>(coldLength.v());


    std::cout << "\nHeat-equation parameters\n";
    std::cout << "----------------------------------------\n";

    std::cout
        << "dt            : "
        << dt.v()
        << " s\n";

    std::cout
        << "alpha         : "
        << alpha.v()
        << " m^2/s\n";

    std::cout
        << "alphaNum      : "
        << alphaNum
        << "\n";


    std::cout
        << "nu            : "
        << nu.v()
        << " m^2/s\n";

    std::cout
        << "nuNum         : "
        << nuNum
        << "\n";

    // ============================================================
    // 4. COMPUTATIONAL DOMAIN
    // ============================================================

    AVec<int> size(
        makeAVec(
            static_cast<int>(nx.v()),
            static_cast<int>(ny.v()),
            static_cast<int>(nz.v())
        )
    );


    asl::Block block(
        size,
        static_cast<double>(dx.v())
    );


    const double Lx =
        static_cast<double>(nx.v() - 1) *
        static_cast<double>(dx.v());
    
    const double Ly =
        static_cast<double>(ny.v() - 1) *
        static_cast<double>(dx.v());
    
    const double Lz =
        static_cast<double>(nz.v() - 1) *
        static_cast<double>(dx.v());

    // Validate future thermal-zone dimensions
    if (
        2.0 *
        (
            static_cast<double>(coldLength.v()) +
            static_cast<double>(transitionLength.v())
        )
        >= Lx
    )
    {
        std::cerr
            << "ERROR: coldLength + transitionLength "
            << "is too large for this heat-pipe length.\n";

        return 1;
    }


    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "          HEAT PIPE MODEL\n";
    std::cout << "========================================\n";


    std::cout << "\nComputational domain\n";
    std::cout << "----------------------------------------\n";

    std::cout
        << "Grid          : "
        << nx.v() << " x "
        << ny.v() << " x "
        << nz.v() << "\n";

    std::cout
        << "dx            : "
        << dx.v()
        << " m\n";

    std::cout
        << "Domain        : "
        << Lx << " x "
        << Ly << " x "
        << Lz
        << " m\n";


    // ============================================================
    // 5. CYLINDRICAL HEAT-PIPE GEOMETRY
    //
    // Cylinder axis is along x.
    // ============================================================

    std::cout
        << "\nGenerating heat-pipe geometry... "
        << std::flush;


    const AVec<double> pipeDirection =
        makeAVec(
            Lx,
            0.0,
            0.0
        );


    const AVec<double> pipeCenter =
        makeAVec(
            0.5 * Lx,
            0.5 * Ly,
            0.5 * Lz
        );


    auto xLeft =
    asl::generatePFLinear(
        makeAVec(1.0, 0.0, 0.0),
        -Lcold
    );

    auto xRight =
    asl::generatePFLinear(
        makeAVec(1.0, 0.0, 0.0),
        -(Lx - Lcold)
    );

    auto sLeft = asl::sign(xLeft);
    auto sRight = asl::sign(xRight);

    auto one =
    asl::generatePFConstant(1.0);

    auto quarter =
    asl::generatePFConstant(0.25);

    auto hotMask =
    quarter *
    (one + sLeft) *
    (one - sRight);

    auto coldPF =
    asl::generatePFConstant(
        static_cast<double>(Tcold.v())
    );

    auto deltaTPF =
    asl::generatePFConstant(
        static_cast<double>(Thot.v() - Tcold.v())
    );

    auto wallTemperature =
    coldPF + deltaTPF * hotMask;




    auto pipeGeometry =
        asl::normalize(
            asl::generateDFCylinder(
                static_cast<double>(pipeRadius.v()),
                pipeDirection,
                pipeCenter
            ),
            static_cast<double>(dx.v())
        );


    auto fluidGeometry = -pipeGeometry;

    auto pipeMap =
        asl::generateDataContainerACL_SP<FlT>(
            block,
            1,
            1u
        );


    auto fluidMap =
    asl::generateDataContainerACL_SP<FlT>(
        block,
        1,
        1u
        );

    asl::initData(
        pipeMap,
        pipeGeometry
    );

    asl::initData(
        fluidMap,
        fluidGeometry
    );


    std::cout << "Finished\n";


    // ============================================================
    // 6. ALLOCATE PHYSICAL FIELDS
    // ============================================================


    // ------------------------------------------------------------
    // Temperature in Kelvin
    // ------------------------------------------------------------

    auto temperatureK =
        asl::generateDataContainerACL_SP<FlT>(
            block,
            1,
            1u
        );


    // Initial condition:
    //
    // T(x,y,z,t=0) = Tcold everywhere
    //
    // The gas starts cold.
    asl::initData(
        temperatureK,
        static_cast<double>(Tcold.v())
    );


    // ------------------------------------------------------------
    // Temperature in Celsius
    //
    // This is currently only for visualization.
    // Later we'll update it from temperatureK.
    // ------------------------------------------------------------

    auto temperatureC =
        asl::generateDataContainerACL_SP<FlT>(
            block,
            1,
            1u
        );


    asl::initData(
        temperatureC,
        static_cast<double>(Tcold.v()) - 273.15
    );


    // ------------------------------------------------------------
    // Argon pressure
    // ------------------------------------------------------------

    auto pAr =
        asl::generateDataContainerACL_SP<FlT>(
            block,
            1,
            1u
        );


    // ------------------------------------------------------------
    // Argon number density
    // ------------------------------------------------------------

    auto nAr =
        asl::generateDataContainerACL_SP<FlT>(
            block,
            1,
            1u
        );


    // ------------------------------------------------------------
    // Argon mass density
    // ------------------------------------------------------------

    auto rhoAr =
        asl::generateDataContainerACL_SP<FlT>(
            block,
            1,
            1u
        );


    // ------------------------------------------------------------
    // Lithium atomic number density
    // ------------------------------------------------------------

    auto nLi =
        asl::generateDataContainerACL_SP<FlT>(
            block,
            1,
            1u
        );


    // ------------------------------------------------------------
    // Li2 molecular number density
    // ------------------------------------------------------------

    auto nLi2 =
        asl::generateDataContainerACL_SP<FlT>(
            block,
            1,
            1u
        );


    // ------------------------------------------------------------
    // Gas velocity
    //
    // u = (ux, uy, uz)
    // ------------------------------------------------------------

    auto velocity =
        asl::generateDataContainerACL_SP<FlT>(
            block,
            3,
            1u
        );

    
    auto advectionVelocity =
    asl::generateDataContainerACL_SP<FlT>(
        block,
        3,
        1u
    );


    // ============================================================
    // 7. INITIAL CONDITIONS
    // ============================================================


    // ------------------------------------------------------------
    // Argon pressure
    //
    // For our first model:
    //
    // P_Ar = constant everywhere.
    // ------------------------------------------------------------

    asl::initData(
        pAr,
        static_cast<double>(pArTorr.v())
    );

    asl::initData(
    advectionVelocity,
    makeAVec(
        0.0,
        0.0,
        0.0
    )
    );


    // Convert pressure to SI
    const double pressurePa =
        static_cast<double>(pArTorr.v()) *
        torrToPa;


    // ------------------------------------------------------------
    // Initial argon number density
    //
    //         P
    // n = --------
    //      k_B T
    //
    // Initially T = Tcold everywhere.
    // ------------------------------------------------------------

    const double nArInitial =
        pressurePa
        /
        (
            kB *
            static_cast<double>(Tcold.v())
        );


    asl::initData(
        nAr,
        nArInitial
    );


    // ------------------------------------------------------------
    // Initial argon mass density
    //
    //          P M
    // rho = --------
    //          R T
    // ------------------------------------------------------------

    const double rhoArInitial =
        (
            pressurePa *
            molarMassAr
        )
        /
        (
            R *
            static_cast<double>(Tcold.v())
        );


    asl::initData(
        rhoAr,
        rhoArInitial
    );


    // ------------------------------------------------------------
    // Lithium initially absent
    // ------------------------------------------------------------

    asl::initData(
        nLi,
        0.0
    );


    asl::initData(
        nLi2,
        0.0
    );


    // ------------------------------------------------------------
    // Gas initially stationary
    // ------------------------------------------------------------

    asl::initData(
        velocity,
        makeAVec(
            0.0,
            0.0,
            0.0
        )
    );


    // ============================================================
    // 8. FUTURE WALL-TEMPERATURE REGION LOCATIONS
    //
    // IMPORTANT:
    //
    // These are NOT being imposed on temperatureK yet.
    //
    // In the next step they will define boundary conditions.
    // ============================================================

    const double x1 =
        static_cast<double>(
            coldLength.v()
        );


    const double x2 =
        x1 +
        static_cast<double>(
            transitionLength.v()
        );


    const double x4 =
        Lx -
        static_cast<double>(
            coldLength.v()
        );


    const double x3 =
        x4 -
        static_cast<double>(
            transitionLength.v()
        );


    std::cout << "\nFuture wall thermal zones\n";
    std::cout << "----------------------------------------\n";

    std::cout
        << "Left cold       : 0 -> "
        << x1
        << " m\n";

    std::cout
        << "Heating region  : "
        << x1 << " -> "
        << x2
        << " m\n";

    std::cout
        << "Hot zone        : "
        << x2 << " -> "
        << x3
        << " m\n";

    std::cout
        << "Cooling region  : "
        << x3 << " -> "
        << x4
        << " m\n";

    std::cout
        << "Right cold      : "
        << x4 << " -> "
        << Lx
        << " m\n";


    // ============================================================
    // 9. PRINT INITIAL PHYSICAL CONDITIONS
    // ============================================================

    std::cout << "\nTemperatures\n";
    std::cout << "----------------------------------------\n";

    std::cout
        << "Initial gas T    : "
        << Tcold.v()
        << " K = "
        << Tcold.v() - 273.15f
        << " C\n";

    std::cout
        << "Future hot wall  : "
        << Thot.v()
        << " K = "
        << Thot.v() - 273.15f
        << " C\n";


    // ------------------------------------------------------------
    // These are only reference values.
    //
    // nArHot is NOT currently placed into the field.
    // ------------------------------------------------------------

    const double nArCold =
        pressurePa
        /
        (
            kB *
            static_cast<double>(Tcold.v())
        );


    const double nArHot =
        pressurePa
        /
        (
            kB *
            static_cast<double>(Thot.v())
        );


    std::cout << "\nArgon initial state\n";
    std::cout << "----------------------------------------\n";

    std::cout
        << "Pressure         : "
        << pArTorr.v()
        << " Torr\n";


    std::cout
        << std::scientific
        << std::setprecision(6);


    std::cout
        << "n_Ar initial     : "
        << nArInitial
        << " m^-3\n";


    std::cout
        << "rho_Ar initial   : "
        << rhoArInitial
        << " kg/m^3\n";


    std::cout << "\nReference values if constant P is assumed:\n";

    std::cout
        << "n_Ar at Tcold    : "
        << nArCold
        << " m^-3\n";

    std::cout
        << "n_Ar at Thot     : "
        << nArHot
        << " m^-3\n";


    std::cout << std::defaultfloat;


    // ===========================================================
    // solving heat-equation
    // ===========================================================
    auto templ(&asl::d3q15());
    asl::SPLBGK lbgk(new asl::LBGK(block,acl::generateVEConstant(FlT(nuNum)), &asl::d3q15()));


    auto heatSolver =generateFDAdvectionDiffusion(temperatureK,alphaNum,advectionVelocity,templ);
    
 
    heatSolver->init();
    lbgk->init();

    asl::SPLBGKUtilities lbgkUtil(new asl::LBGKUtilities(lbgk));

    lbgkUtil->initF(acl::generateVEConstant(0.0,0.0, 0.0 ));



    std::vector<asl::SlicesNames> outerWalls = {
    asl::X0, asl::XE,
    asl::Y0, asl::YE,
    asl::Z0, asl::ZE
    };

    std::vector<asl::SlicesNames> endCaps = {
    asl::X0,
    asl::XE
    };

    auto coldOuterBC =
    asl::generateBCConstantValue(
        temperatureK,
        static_cast<double>(Tcold.v()),
        outerWalls
    );

    auto coldEndBC =
    asl::generateBCConstantValue(
        temperatureK,
        static_cast<double>(Tcold.v()),
        endCaps
    );

    auto hotPipeBC =
    asl::generateBCConstantValue(
        temperatureK,
        static_cast<double>(Thot.v()),
        pipeMap
    );

    auto wallTemperatureBC =
    asl::generateBCConstantValue(
        temperatureK,
        wallTemperature,
        fluidMap
    );


    auto pipeNoSlipBC =
    asl::generateBCNoSlip(
        lbgk,
        fluidMap
    );
    auto pipeNoSlipVel =
    asl::generateBCNoSlipVel(
        lbgk,
        fluidMap
    );

    auto outerNoSlipBC =
    asl::generateBCNoSlip(
        lbgk,
        {
            asl::X0, asl::XE,
            asl::Y0, asl::YE,
            asl::Z0, asl::ZE
        }
    );

    coldOuterBC->init();
    //coldEndBC->init();

    //confining the gas in cylindrical interior region.
    pipeNoSlipBC->init();
    pipeNoSlipVel->init();
    outerNoSlipBC->init();

    wallTemperatureBC->init();
    //hotPipeBC->init();

    // Apply BC BEFORE the first heat-equation step
    coldOuterBC->execute();

    wallTemperatureBC->execute();
    //hotPipeBC->execute();

    pipeNoSlipBC->execute();
    outerNoSlipBC->execute();





    //coldEndBC->execute();
    //pipeNoSlipVel->execute();




    std::cout
        << "\nWriting VTK output... "
        << std::flush;


    asl::WriterVTKXML writer(
        "heatpipe_simulation"
    );


    // Geometry
    writer.addScalars(
        "pipe",
        *pipeMap
    );

    writer.addScalars(
        "rho_LB",
        *lbgk->getRho()
    );

    writer.addVector(
        "u_LB",
        *lbgk->getVelocity()
    );

    // Temperature
    writer.addScalars(
        "T_K",
        *temperatureK
    );


    writer.addScalars(
        "T_C",
        *temperatureC
    );


    // Argon
    writer.addScalars(
        "P_Ar_Torr",
        *pAr
    );

    writer.addScalars(
        "fluid",
        *fluidMap
    );

    writer.addScalars(
        "n_Ar",
        *nAr
    );


    writer.addScalars(
        "rho_Ar",
        *rhoAr
    );


    // Lithium
    writer.addScalars(
        "n_Li",
        *nLi
    );


    writer.addScalars(
        "n_Li2",
        *nLi2
    );


    writer.addVector(
        "u_adv",
        *advectionVelocity
    );

    // Velocity
    //writer.addVector(
    //    "u",
    //    *velocity
    //);



    for (unsigned int i = 1; i <= 100; ++i)
    {
    // --------------------------------------------------------
    // 1. Fluid step
    // --------------------------------------------------------

    lbgk->execute();

    pipeNoSlipBC->execute();
    outerNoSlipBC->execute();

    pipeNoSlipVel->execute();


    // --------------------------------------------------------
    // 2. MASKING BLOCK
    //    Copy u_LB -> u_adv only inside fluidMap
    // --------------------------------------------------------

    {
        auto& fluidContainer =
            fluidMap->getContainer();

        auto& lbVelocityContainer =
            lbgk->getVelocity()->getContainer();

        auto& advVelocityContainer =
            advectionVelocity->getContainer();


        auto fluidPtr =
            acl::map<FlT>(
                fluidContainer[0]
            );


        auto lbUx =
            acl::map<FlT>(
                lbVelocityContainer[0]
            );

        auto lbUy =
            acl::map<FlT>(
                lbVelocityContainer[1]
            );

        auto lbUz =
            acl::map<FlT>(
                lbVelocityContainer[2]
            );


        auto advUx =
            acl::map<FlT>(
                advVelocityContainer[0]
            );

        auto advUy =
            acl::map<FlT>(
                advVelocityContainer[1]
            );

        auto advUz =
            acl::map<FlT>(
                advVelocityContainer[2]
            );


        const auto& fullSize =
            advectionVelocity->getBlock().getSize();


        const size_t numberOfNodes =
            static_cast<size_t>(fullSize[0]) *
            static_cast<size_t>(fullSize[1]) *
            static_cast<size_t>(fullSize[2]);


        for (size_t j = 0; j < numberOfNodes; ++j)
        {
            if (fluidPtr.get()[j] > 0.1f)
            {
                advUx.get()[j] = lbUx.get()[j];
                advUy.get()[j] = lbUy.get()[j];
                advUz.get()[j] = lbUz.get()[j];
            }
            else
            {
                advUx.get()[j] = 0.0f;
                advUy.get()[j] = 0.0f;
                advUz.get()[j] = 0.0f;
            }
        }
    }


    // --------------------------------------------------------
    // 3. Temperature step
    // --------------------------------------------------------

    heatSolver->execute();

    wallTemperatureBC->execute();


    auto& TContainer =
    temperatureK->getContainer();

    auto& fluidContainer =
        fluidMap->getContainer();
    
    auto Tptr =
        acl::map<FlT>(
            TContainer[0]
        );
    
    auto fluidPtr =
        acl::map<FlT>(
            fluidContainer[0]
        );


    const auto& fullSize =
    temperatureK->getBlock().getSize();

    const size_t numberOfNodes =
        static_cast<size_t>(fullSize[0]) *
        static_cast<size_t>(fullSize[1]) *
        static_cast<size_t>(fullSize[2]);


    double sumInverseT = 0.0;
    size_t fluidNodes = 0;


    for (size_t j = 0; j < numberOfNodes; ++j)
    {
    if (fluidPtr.get()[j] > 0.1f)
    {
        const double T =
            static_cast<double>(
                Tptr.get()[j]
            );

        sumInverseT += 1.0 / T;
        ++fluidNodes;
    }
    }


    const double meanInverseT =
    sumInverseT /
    static_cast<double>(fluidNodes);



    const double pressureArNow =
    pressurePa *
    (1.0 / static_cast<double>(Tcold.v())) /
    meanInverseT;

    auto& nArContainer =
    nAr->getContainer();

    auto nArPtr =
    acl::map<FlT>(
        nArContainer[0]
    );


    for (size_t j = 0; j < numberOfNodes; ++j)
    {
    if (fluidPtr.get()[j] > 0.1f)
    {
        const double T =
            static_cast<double>(
                Tptr.get()[j]
            );

        const double nLocal =
            pressureArNow /
            (
                kB * T
            );

        nArPtr.get()[j] =
            static_cast<FlT>(
                nLocal
            );
    }
    else
    {
        nArPtr.get()[j] = 0.0f;
    }
    }
    // --------------------------------------------------------
    // 4. Output
    // --------------------------------------------------------

    if (!(i % 10))
    {
        pipeNoSlipVel->execute();

        std::cout
            << "Writing iteration "
            << i
            << std::endl;

        std::cout
        << "Iteration " << i
        << "  P_Ar = "
        << pressureArNow / torrToPa
        << " Torr"
        << std::endl;

        writer.write();
    }
    }



    


    //writer.write();


    std::cout << "Finished\n";


    // ============================================================
    // 11. FINISH
    // ============================================================

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Initial state prepared successfully.\n";
    std::cout << "Output: heatpipe_initial_0.vti\n";
    std::cout << "========================================\n";


    std::cout << "\nNEXT STEP:\n";
    std::cout
        << "Create FDAdvectionDiffusion for T_K and solve\n"
        << "dT/dt = alpha * Laplacian(T).\n\n";


    return 0;
}