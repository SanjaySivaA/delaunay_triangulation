#!/bin/bash

# run after cloning PBBS and initializing its parlaylib submodule

set -e

pbbs="$HOME/pbbsbench"
output="$HOME/pbbs-points"
mkdir -p "$output"

make -C "$pbbs/testData/geometryData" randPoints
rand="$pbbs/testData/geometryData/randPoints"

for n in 1000000 10000000; do
    "$rand" -d 2 "$n" "$output/square_${n}.2dpoints"
    "$rand" -k -d 2 "$n" "$output/kuzmin_${n}.2dpoints"
done

echo "PBBS inputs are in $output"