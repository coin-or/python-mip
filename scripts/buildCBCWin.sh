PREFIX="x86_64-w64-mingw32-"
export CC=${PREFIX}gcc
export FC=${PREFIX}gfortran
export CXX=${PREFIX}c++
export LD=${PREFIX}ld
export AR=${PREFIX}ar
export AS=${PREFIX}as
export NM=${PREFIX}nm
export STRIP=${PREFIX}strip
export RANLIB=${PREFIX}ranlib
export DLLTOOL=${PREFIX}dlltool
export OBJDUMP=${PREFIX}objdump
export RESCOMP=${PREFIX}windres
export MINGWROOT=/usr/x86_64-w64-mingw32/

export CXXFLAGS="-Ofast -fPIC -flto -m64 -DNDEBUG -fprefetch-loop-arrays -static"
export CFLAGS="-Ofast -fPIC -flto -m64 -DNDEBUG -fprefetch-loop-arrays -static"
export FCFLAGS="-Ofast -fPIC -flto -m64 -DNDEBUG -fprefetch-loop-arrays -static"
export F77FLAGS="-Ofast -fPIC -flto -m64 -DNDEBUG -fprefetch-loop-arrays -static"
export LDFLAGS="-Ofast -static -fPIC -flto -m64 -Bstatic -lgfortran -static-libgcc -static-libgfortran -static-libstdc++"

cd ~/src/cbctrunk/

$(CXX) \
    -I/opt/w64/include/coin/ -L/opt/w64/lib/coin/ \
    -I${MINGWROOT}/include/ $(CXXFLAGS) \
    -mdll -shared -DCBC_THREAD ~/src/cbctrunk/Cbc/src/Cbc_C_Interface.cpp $(LDFLAGS) \
    -o ~/git/mip/mip/libraries/cbc-c-windows-x86-64.dll

