#!/usr/bin/bash

DIR=$(pwd)
APP="apps"
BC="boundary_conditions"

# applications
## calculation of local and global Sherwood numbers
cd ${APP}/calcSh/ && wclean && wmake && cd $DIR
## incompressible solver with additional solution of scalar transport equations
cd ${APP}apps/scalarPimpleFoam/ && wclean && wmake && cd $DIR

# boundary conditions
## inlet velocity
cd ${BC}/ptInletVelocity/ && wclean && wmake && cd $DIR
## bubble surface velocity
cd ${BC}/ptSurfaceVeloctiy/ && wclean && wmake && cd $DIR
## bubble surface displacement
cd ${BC}/ptBoundaryDisplacement/ && wclean && wmake && cd $DIR