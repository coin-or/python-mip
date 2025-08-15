export CFLAGS="-Ofast -fPIC -flto -DNDEBUG -fprefetch-loop-arrays -I/opt/gcc/include/"
export FFLAGS="-Ofast -fPIC -flto -DNDEBUG -I/opt/gcc/include/"
export CXXFLAGS="-Ofast -fPIC -flto -fprefetch-loop-arrays -DNDEBUG -I/opt/gcc/include/"
export LDFLAGS="-Ofast -fPIC -L/opt/gcc/lib -flto -static-libgcc -static-libstdc++ -static-libgfortran"

dir=`pwd`
mkdir -p ~/prog
cd ~/prog
IDIR=`pwd`
export PKG_CONFIG_PATH=${IDIR}/lib/pkgconfig/:${PKG_CONFIG_PATH}

#cd $dir/ThirdParty-Metis
#./configure --prefix=$IDIR --enable-cbc-parallel --enable-static --disable-shared --enable-gnu-packages
#make -j 6
#make -j 6 install

cd $dir/ThirdParty-Blas
./configure --prefix=$IDIR --enable-cbc-parallel --enable-static --disable-shared --enable-gnu-packages
make -j 6
make -j 6 install

cd $dir/ThirdParty-Lapack
./configure --prefix=$IDIR --enable-cbc-parallel --enable-static --disable-shared --enable-gnu-packages
make -j 6
make -j 6 install

#cd $dir/ThirdParty-Mumps
#./configure --prefix=$IDIR --enable-cbc-parallel --enable-static --disable-shared --enable-gnu-packages
#make -j 6
#make -j 6 install

cd $dir/ThirdParty-Glpk
./configure --prefix=$IDIR --enable-cbc-parallel --enable-static --disable-shared --enable-gnu-packages
make -j 6
make -j 6 install

cd $dir/CoinUtils
./configure --prefix=$IDIR --enable-cbc-parallel --enable-static --disable-shared --enable-gnu-packages
make -j 6
make -j 6 install

cd $dir/Osi
./configure --prefix=$IDIR --enable-cbc-parallel --enable-static --disable-shared --enable-gnu-packages
make -j 6
make -j 6 install

cd $dir/Clp
./configure --prefix=$IDIR --enable-cbc-parallel --enable-static --disable-shared --enable-gnu-packages
make -j 6
make -j 6 install

cd $dir/Cgl
./configure --prefix=$IDIR --enable-cbc-parallel --enable-static --disable-shared --enable-gnu-packages
make -j 6
make -j 6 install


cd $dir/Cbc
./configure --prefix=$IDIR --enable-cbc-parallel --enable-static --disable-shared --enable-gnu-packages
make -j 6
make -j 6 install

cd $IDIR

patch -p1 < $dir/../scripts/coinutils-glpk.patch

cd $dir

g++ -shared -Ofast -fPIC -o ../mip/libraries/cbc-c-linux-aarch64.so \
-I${IDIR}/include/coin-or/ \
 -DCBC_THREAD \
 ./Cbc/src/Cbc_C_Interface.cpp \
 -L/opt/gcc/lib64/ -L${IDIR}/lib/ \
 -lCbc -lpthread -lrt -lCgl -lOsiClp -lClp -lOsi -lCoinUtils -lreadline -lz -lbz2 \
 -lcoinlapack -lcoinblas -lgfortran -lm -static-libgcc -static-libstdc++ -static-libgfortran -lcoinglpk
