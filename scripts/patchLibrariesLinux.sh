#! /bin/sh
#
# patchLibraries.sh
# Copyright (C) 2020 haroldo <haroldo@soyuz>
#
# Distributed under terms of the MIT license.
#


for lib in libCbcSolver.so libcholmod.so.1.7.1 libClp.so libCoinUtils.so libOsi.so libOsiClp.so libCbc.so libCgl.so libClpSolver.so libamd.so.2.2.0 libcholmod.so.1.7.1 libblas.so.3gf libamd.so.2.2.0 libcoinmumps.so libcoinglpk.so libcoinasl.so libreadline.so.6 liblapack.so.3gf  libblas.so.3gf;
do
    echo patching "$lib"
    #patchelf --set-rpath ./ ../mip/libraries/lin64/$lib
    echo $lib
    chrpath -r $lib
done

if [ -f ../mip/libraries/lin64/cbc.bin ];
then
    chrpath -r ./ ../mip/libraries/lin64/cbc.bin

    #patchelf --set-rpath ./ ../mip/libraries/lin64/cbc.bin
fi
