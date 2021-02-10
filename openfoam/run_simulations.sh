#!/usr/bin/bash


RUN=$(pwd)/run
mkdir -p $RUN

meshes=("blockMesh_ref_0" "blockMesh_ref_1" "blockMesh_ref_2")
cases=("CB4_ref_0" "CB4_ref_1" "CB1_ref_1" "CB2_ref_1" "CB3_ref_1" "CB4_ref_2_parallel" "CB4_ref_1_slip" "CB4_ref_1_frozen")

# create meshes
for mesh in ${meshes[@]}; do
  target=${RUN}/${mesh}
  if [ -d  $target ]; then
    echo "folder ${target} already exists. Skipping case..."
  else
    cp -r test_cases/$mesh ${RUN}/
    ${RUN}/${mesh}/Allrun
  fi
done

for case in ${cases[@]}; do
  target=${RUN}/${case}
  if [ -d  $target ]; then
    echo "folder ${target} already exists. Skipping case..."
  else
    cp -r test_cases/$case ${RUN}/
    ${RUN}/${case}/Allrun &
  fi
done
