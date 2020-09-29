/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::ptInletOutletVelocityFvPatchVectorField

Group
    grpInletBoundaryConditions

Description
    This boundary condition provides an inflow condition for velocity.
    The inflow velocity is computed based on a PyTorch model. In case
    of outflow, the gradient is set ot zero.

Usage
    \table
        Property     | Description             | Required    | Default value
        U            | velocity field name     | no          | U
        model        | pyTorch model name      | yes         |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            ptInletOutletVelocity;
        U               U;
        model           "rise_vel_CB1.ts"
    }
    \endverbatim

See also
    Foam::inletOutletFvPatchField

SourceFiles
    ptInletOutletVelocityFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef inletOutletVelocityFvPatchVectorField_H
#define inletOutletVelocityFvPatchVectorField_H

#include <torch/script.h>
#include "inletOutletFvPatchFields.H"
#include "Time.H"
#include "polyMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
       Class ptInletOutletVelocityFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class ptInletOutletVelocityFvPatchVectorField
:
    public inletOutletFvPatchVectorField
{
    // Private data

        //- Name of the velocity field
        word UName_;

        //- Name of the PyTorch model
        word ptModelName_;

        //- PyTorch model
        torch::jit::script::Module velocity_model_;

public:

    //- Runtime type information
    TypeName("ptInletOutletVelocity");


    // Constructors

        //- Construct from patch and internal field
        ptInletOutletVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        ptInletOutletVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given
        //  ptInletOutletVelocityFvPatchVectorField onto a new patch
        ptInletOutletVelocityFvPatchVectorField
        (
            const ptInletOutletVelocityFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        ptInletOutletVelocityFvPatchVectorField
        (
            const ptInletOutletVelocityFvPatchVectorField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new ptInletOutletVelocityFvPatchVectorField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        ptInletOutletVelocityFvPatchVectorField
        (
            const ptInletOutletVelocityFvPatchVectorField&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchVectorField> clone
        (
            const DimensionedField<vector, volMesh>& iF
        ) const
        {
            return tmp<fvPatchVectorField>
            (
                new ptInletOutletVelocityFvPatchVectorField(*this, iF)
            );
        }


    // Member functions

        // Evaluation functions

            //- Update the coefficients associated with the patch field
            virtual void updateCoeffs();


        //- Write
        virtual void write(Ostream&) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
