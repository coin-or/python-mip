export MACOSX_DEPLOYMENT_TARGET="10.7"
export CFLAGS="-fPIC -Ofast -DNDEBUG -ffast-math -mmacosx-version-min=10.7"
export CXXFLAGS="-fPIC -Og -DNDEBUG -ffast-math -std=c++11 -stdlib=libc++ -mmacosx-version-min=10.7"
export F77FLAGS="-fPIC -Ofast -DNDEBUG -ffast-math"
export LDFLAGS="-fPIC -Ofast -DNDEBUG -ffast-math"

DIR=`pwd`
OUTDIR="/opt/cbc/bin"
export PKG_CONFIG_PATH="${OUTDIR}/lib/pkgconfig/:${PKG_CONFIG_PATH}"

echo
echo "Making and installing Glpk"
cd ${DIR}/ThirdParty-Glpk
./configure --prefix=${OUTDIR}/ --enable-static --disable-shared --without-glpk
git pull
make
make install

echo
echo "Making and installing Lapack"
cd ${DIR}/ThirdParty-Lapack
./configure --prefix=${OUTDIR}/ --enable-static --disable-shared --without-glpk
git pull
make
make install

echo
echo "Making and installing Blas"
cd ${DIR}/ThirdParty-Blas
./configure --prefix=${OUTDIR}/ --enable-static --disable-shared --without-glpk
git pull
make
make install

echo
echo "Making and installing CoinUtils"
cd ${DIR}/CoinUtils
./configure --prefix=${OUTDIR}/ --enable-static --disable-shared --without-glpk
git pull
make
make install

echo
echo "Making and installing Osi"
cd ${DIR}/Osi
./configure --prefix=${OUTDIR}/ --enable-static --disable-shared --without-glpk
git pull
make
make install

echo
echo "Making and installing Clp"
cd ${DIR}/Clp
./configure --prefix=${OUTDIR}/ --enable-static --disable-shared --without-glpk
git pull
make
make install

echo
echo "Making and installing Cgl"
cd ${DIR}/Cgl
./configure --prefix=${OUTDIR}/ --enable-static --disable-shared --without-glpk
git pull
make
make install

echo
echo "Making and installing Cbc"
cd ${DIR}/Cbc
./configure --prefix=${OUTDIR}/ --enable-cbc-parallel --enable-static --disable-shared --without-glpk
git pull
make
make install

echo
echo "Compiling dynamic library"
cd ${DIR}
clang++ -shared -Ofast -fPIC -o cbc-c-darwin-x86-64.dylib \
        -I${OUTDIR}/include/coin-or/ -I${OUTDIR}/include/coin -L${OUTDIR}/lib \
        ./Cbc/src/Cbc_C_Interface.cpp \
        -lCbcSolver -lCbc -lCgl -lOsiClp -lClpSolver -lClp -lOsi -lCoinUtils \
        -lbz2 -lz -llapack ${CXXFLAGS} -stdlib=libc++ -lreadline 
echo "Done!"
