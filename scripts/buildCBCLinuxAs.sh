export CFLAGS="-Og -fPIC -g -fsanitize=address"
export FFLAGS="-Og -fPIC -g -fsanitize=address"
export CXXFLAGS="-Og -fPIC -g -fsanitize=address"
export LDFLAGS="-Og -fPIC -g -static-libgcc -static-libstdc++ -fsanitize=address"

dir=`pwd`
mkdir -p ~/prog
cd ~/prog
IDIR=`pwd`
export PKG_CONFIG_PATH=${IDIR}/lib/pkgconfig/:${PKG_CONFIG_PATH}

cd $dir/CoinUtils
./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas  --enable-static --disable-shared 
make -j 6
make -j 6 install

cd $dir/Osi
./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas  --enable-static --disable-shared 
make -j 6
make -j 6 install

cd $dir/Clp
./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas  --enable-static --disable-shared 
make -j 6
make -j 6 install

cd $dir/Cgl
./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas  --enable-static --disable-shared 
make -j 6
make -j 6 install


cd $dir/Cbc
./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas  --enable-static --disable-shared 
make -j 6
make -j 6 install

cd $dir

