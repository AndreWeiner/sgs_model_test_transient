/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2017-2020 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

#include "ptInletOutletVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::ptInletOutletVelocityFvPatchVectorField::
ptInletOutletVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    inletOutletFvPatchVectorField(p, iF),
    UName_("U"),
    ptModelName_("model"),
    velocity_model_(torch::jit::load(ptModelName_))
{
    this->refValue() = Zero;
    this->refGrad() = Zero;
    this->valueFraction() = 0.0;
}


Foam::ptInletOutletVelocityFvPatchVectorField::
ptInletOutletVelocityFvPatchVectorField
(
    const ptInletOutletVelocityFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    inletOutletFvPatchVectorField(ptf, p, iF, mapper),
    UName_(ptf.UName_),
    ptModelName_(ptf.ptModelName_),
    velocity_model_(ptf.velocity_model_)
{}


Foam::ptInletOutletVelocityFvPatchVectorField::
ptInletOutletVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    inletOutletFvPatchVectorField(p, iF),
    UName_(dict.getOrDefault<word>("U", "U")),
    ptModelName_(dict.get<word>("model")),
    velocity_model_(torch::jit::load(ptModelName_))
    {
    this->patchType() = dict.getOrDefault<word>("patchType", word::null);

    vector v_in(0.0, -0.1, 0.0);
    if (dict.found("value"))
    {
        fvPatchField<vector>::operator=
        (
            vectorField("value", dict, p.size())
        );
    }
    else
    {
        fvPatchField<vector>::operator=(this->refValue());
    }

    this->refValue() = Zero;
    this->refGrad() = Zero;
    this->valueFraction() = 1.0;
}


Foam::ptInletOutletVelocityFvPatchVectorField::
ptInletOutletVelocityFvPatchVectorField
(
    const ptInletOutletVelocityFvPatchVectorField& tppsf
)
:
    inletOutletFvPatchVectorField(tppsf),
    UName_(tppsf.UName_),
    ptModelName_(tppsf.ptModelName_),
    velocity_model_(tppsf.velocity_model_)
{}


Foam::ptInletOutletVelocityFvPatchVectorField::
ptInletOutletVelocityFvPatchVectorField
(
    const ptInletOutletVelocityFvPatchVectorField& tppsf,
    const DimensionedField<vector, volMesh>& iF
)
:
    inletOutletFvPatchVectorField(tppsf, iF),
    UName_(tppsf.UName_),
    ptModelName_(tppsf.ptModelName_),
    velocity_model_(tppsf.velocity_model_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::ptInletOutletVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const fvsPatchField<scalar>& phip =
        patch().lookupPatchField<surfaceScalarField, scalar>("phi");

    const polyMesh& mesh = this->internalField().mesh();
    const Time& t = mesh.time();

    auto time = torch::zeros({1, 1}, torch::kFloat64);
    time[0][0] = t.value();
    std::vector<torch::jit::IValue> feature{time};
    auto velocity = velocity_model_.forward(feature).toTensor();
    auto velocityAcc = velocity.accessor<double, 2>();
    //Info << "Setting inlet velocity to " << velocityAcc[0][0] << "\n";

    //scalar gM1ByG = (gamma_ - 1.0)/gamma_;
    vector v_in(0.0, -velocityAcc[0][0], 0.0);

    this->refValue() = v_in;
    //    T0_/(1.0 + 0.5*psip*gM1ByG*(1.0 - pos0(phip))*magSqr(Up));
    //if (t.value() < 1.0e-3) {
    //    this->valueFraction() = 1.0;
    //} else {
        this->valueFraction() = 1.0 - pos0(phip);
    //}
    //std::cout << max(this->valueFraction()) << "\n";
    vectorField::operator=(
        this->valueFraction() * v_in +
        (1.0 - this->valueFraction()) * this->patchInternalField()
    );

 //   inletOutletFvPatchVectorField::updateCoeffs();
}


void Foam::ptInletOutletVelocityFvPatchVectorField::write(Ostream& os)
const
{
    fvPatchVectorField::write(os);
    os.writeEntryIfDifferent<word>("U", "U", UName_);
    os.writeEntryIfDifferent<word>("model", "model", ptModelName_);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
        ptInletOutletVelocityFvPatchVectorField
    );
}

// ************************************************************************* //
