clear;
clc;
close all;

%% ============================================================
% USER PARAMETERS
% =============================================================

% Temperatures
Tmin = 19 + 273.15;       % K
Tmax = 588 + 273.15;      % K

% ------------------------------------------------------------
% Main longitudinal tube
% ------------------------------------------------------------

Rmain = 15;               % mm
zmin  = -150;             % mm
zmax  =  150;             % mm

% Temperature falloff along main z-axis
zHot = 110;               % mm
wz   = 15;                % mm

% ------------------------------------------------------------
% Two lateral arms along +x and -x
% ------------------------------------------------------------

Rarm = 10;                % mm
Larm = 50;                % mm

% Main tube outer radial location where arm starts
xArmStart = Rmain;

% Arm outer end
xArmEnd = Rmain + Larm;

% Temperature falloff along lateral arms
%
% xHot is measured from x = 0.
% Larger xHot => hot region extends farther down arms.
%
xHot = 30;                % mm

% Larger wx => smoother temperature transition.
wx = 8;                   % mm


%% ============================================================
% RESOLUTION FOR POINT CLOUD
% =============================================================

Nz = 250;
Nx = 120;

NthetaMain = 80;
NthetaArm  = 60;


%% ============================================================
% MAIN CYLINDER SURFACE
%
% Main cylinder axis is z.
%
% x = R cos(theta)
% y = R sin(theta)
% =============================================================

z = linspace(zmin, zmax, Nz);

thetaMain = linspace(0, 2*pi, NthetaMain);

[Z, ThetaMain] = meshgrid(z, thetaMain);

X = Rmain .* cos(ThetaMain);
Y = Rmain .* sin(ThetaMain);


%% ============================================================
% TEMPERATURE FUNCTION ALONG z
%
% fz ~ 1 in central hot region
% fz ~ 0 at cold ends
%
%
%          1
% fz = -------- [
%          2
%
%        tanh((z + zHot)/wz)
%
%          -
%
%        tanh((z - zHot)/wz)
%      ]
%
% =============================================================

fz = 0.5 .* ...
    ( ...
        tanh((Z + zHot) ./ wz) ...
        - ...
        tanh((Z - zHot) ./ wz) ...
    );


%% ============================================================
% MAIN CYLINDER WALL TEMPERATURE
% =============================================================

Tmain = Tmin + (Tmax - Tmin) .* fz;


%% ============================================================
% POSITIVE-x LATERAL ARM
%
% Arm axis is +x.
%
% Cross section is in the y-z plane.
% =============================================================

xPos = linspace( ...
    xArmStart, ...
    xArmEnd, ...
    Nx ...
);

thetaArm = linspace(0, 2*pi, NthetaArm);

[Xpos, ThetaPos] = meshgrid( ...
    xPos, ...
    thetaArm ...
);

Ypos = Rarm .* cos(ThetaPos);

Zpos = Rarm .* sin(ThetaPos);


%% ============================================================
% NEGATIVE-x LATERAL ARM
% =============================================================

xNeg = linspace( ...
    -xArmStart, ...
    -xArmEnd, ...
    Nx ...
);

[Xneg, ThetaNeg] = meshgrid( ...
    xNeg, ...
    thetaArm ...
);

Yneg = Rarm .* cos(ThetaNeg);

Zneg = Rarm .* sin(ThetaNeg);


%% ============================================================
% LATERAL ARM TEMPERATURE FUNCTION
%
% Same function for BOTH arms because we use abs(x).
%
%
%               1
% f_arm(x) = -------- [
%               2
%
%             1 -
%
%             tanh(
%                (|x| - xHot) / wx
%             )
%          ]
%
%
% Near the center:
%
%        f_arm ~ 1
%
% Toward either outer arm:
%
%        f_arm -> 0
%
% =============================================================

fArmPos = 0.5 .* ...
    ( ...
        1 ...
        - ...
        tanh( ...
            (abs(Xpos) - xHot) ./ wx ...
        ) ...
    );

fArmNeg = 0.5 .* ...
    ( ...
        1 ...
        - ...
        tanh( ...
            (abs(Xneg) - xHot) ./ wx ...
        ) ...
    );


%% ============================================================
% LATERAL ARM WALL TEMPERATURE
% =============================================================

TarmPos = ...
    Tmin ...
    + ...
    (Tmax - Tmin) .* fArmPos;

TarmNeg = ...
    Tmin ...
    + ...
    (Tmax - Tmin) .* fArmNeg;


%% ============================================================
% CREATE 3D POINT CLOUD
% =============================================================

figure;

% Main longitudinal tube
scatter3( ...
    X(:), ...
    Y(:), ...
    Z(:), ...
    10, ...
    Tmain(:), ...
    'filled' ...
);

hold on;

% Positive-x arm
scatter3( ...
    Xpos(:), ...
    Ypos(:), ...
    Zpos(:), ...
    10, ...
    TarmPos(:), ...
    'filled' ...
);

% Negative-x arm
scatter3( ...
    Xneg(:), ...
    Yneg(:), ...
    Zneg(:), ...
    10, ...
    TarmNeg(:), ...
    'filled' ...
);


%% ============================================================
% PLOT FORMATTING
% =============================================================

axis equal;
grid on;
box on;

xlabel('x [mm]');
ylabel('y [mm]');
zlabel('z [mm]');

title( ...
    sprintf( ...
        'Heat-Pipe Wall Temperature Model, T_{max} = %.0f ^\\circC', ...
        Tmax - 273.15 ...
    ) ...
);

cb = colorbar;

ylabel( ...
    cb, ...
    'Temperature [K]' ...
);

view(3);

rotate3d on;


%% ============================================================
% OPTIONAL: FORCE TEMPERATURE COLOR RANGE
% =============================================================

clim([Tmin Tmax]);


%% ============================================================
% PRINT PARAMETERS
% =============================================================

fprintf('\n');
fprintf('Temperature model parameters\n');
fprintf('-------------------------------------\n');

fprintf( ...
    'Tmin      = %.2f K (%.2f C)\n', ...
    Tmin, ...
    Tmin - 273.15 ...
);

fprintf( ...
    'Tmax      = %.2f K (%.2f C)\n', ...
    Tmax, ...
    Tmax - 273.15 ...
);

fprintf( ...
    'zHot      = %.2f mm\n', ...
    zHot ...
);

fprintf( ...
    'wz        = %.2f mm\n', ...
    wz ...
);

fprintf( ...
    'xHot      = %.2f mm\n', ...
    xHot ...
);

fprintf( ...
    'wx        = %.2f mm\n', ...
    wx ...
);

fprintf('\n');


%% ============================================================
% OPTIONAL: PLOT TEMPERATURE ALONG MAIN z AXIS
% =============================================================

zPlot = linspace(zmin, zmax, 1000);

fzPlot = 0.5 .* ...
    ( ...
        tanh((zPlot + zHot) ./ wz) ...
        - ...
        tanh((zPlot - zHot) ./ wz) ...
    );

TzPlot = ...
    Tmin ...
    + ...
    (Tmax - Tmin) .* fzPlot;

figure;

plot( ...
    zPlot, ...
    TzPlot, ...
    'LineWidth', ...
    2 ...
);

grid on;

xlabel('z [mm]');
ylabel('Wall temperature [K]');

title('Longitudinal Wall Temperature');


%% ============================================================
% OPTIONAL: PLOT TEMPERATURE ALONG THE TWO LATERAL ARMS
% =============================================================

xPlot = linspace( ...
    -(Rmain + Larm), ...
     (Rmain + Larm), ...
    1000 ...
);

fArmPlot = 0.5 .* ...
    ( ...
        1 ...
        - ...
        tanh( ...
            (abs(xPlot) - xHot) ./ wx ...
        ) ...
    );

TxPlot = ...
    Tmin ...
    + ...
    (Tmax - Tmin) .* fArmPlot;

figure;

plot( ...
    xPlot, ...
    TxPlot, ...
    'LineWidth', ...
    2 ...
);

grid on;

xlabel('x [mm]');
ylabel('Wall temperature [K]');

title('Lateral-Arm Wall Temperature');