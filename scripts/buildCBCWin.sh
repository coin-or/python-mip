export CFLAGS="-Ofast -fPIC -flto -DNDEBUG -fprefetch-loop-arrays"
export FFLAGS="-Ofast -fPIC -flto -DNDEBUG"
export CXXFLAGS="-Ofast -fPIC -flto -DNDEBUG -I/home/haroldo/prog/win64/include/coin-or/"
export LDFLAGS="-Ofast -fPIC -flto -static-libgcc -static-libstdc++ -static-libgfortran -L/home/haroldo/prog/win64/lib/"
export PKG_CONFIG_PATH=:/home/haroldo/prog/win64/lib/pkgconfig/

dir=`pwd`
mkdir -p ~/prog
mkdir -p ~/prog/win64
cd ~/prog/win64
IDIR=`pwd`
#export PKG_CONFIG_PATH=${IDIR}/lib/pkgconfig/

#cd $dir/ThirdParty-Glpk
#make clean -j 3
#make distclean -j 3
#./configure --prefix=${IDIR} --enable-cbc-parallel --enable-static --disable-shared --host=x86_64-w64-mingw32 --disable-gnu-packages
#make -j 6
#make -j 6 install


#cd $dir/ThirdParty-Lapack
#make clean -j 3
#make distclean -j 3
#./configure --prefix=${IDIR}  --enable-cbc-parallel --enable-static --disable-shared --host=x86_64-w64-mingw32 --disable-gnu-packages
#make -j 6

#cd $dir/ThirdParty-Blas
#make clean -j 3
#make distclean -j 3
#./configure --prefix=${IDIR}  --enable-cbc-parallel --enable-static --disable-shared --host=x86_64-w64-mingw32 --disable-gnu-packages
#make -j 6
#make -j 6 install

#cd $dir/CoinUtils
#make clean -j 3
#make distclean -j 3
#./configure --prefix=${IDIR}  --enable-cbc-parallel --enable-static --disable-shared --host=x86_64-w64-mingw32 --disable-gnu-packages
#make -j 6
#make -j 6 install


#cd $dir/Osi
#make clean -j 3
#make distclean -j 3
#./configure --prefix=${IDIR} --enable-cbc-parallel --enable-static --disable-shared --host=x86_64-w64-mingw32 --disable-gnu-packages \
#    --with-coinutils --with-coinutils-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-coinutils-lflags="-L/home/haroldo/prog/win64/lib/"
#make -j 6
#make -j 6 install

#cd $dir/Clp
#make clean -j 3
#make distclean -j 3
#./configure --prefix=${IDIR} --enable-cbc-parallel --enable-static --disable-shared --host=x86_64-w64-mingw32 --disable-gnu-packages  \
#    --with-coinutils --with-coinutils-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-coinutils-lflags="-L/home/haroldo/prog/win64/lib/ -lCoinUtils" \
#    --with-osi --with-osi-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-osi-lflags="-L/home/haroldo/prog/win64/lib/ -lOsi" 
#
#make -j 6
#make -j 6 install

#cd $dir/Cgl
#make clean -j 3
#make distclean -j 3
#./configure --prefix=${IDIR} --enable-cbc-parallel --enable-static --disable-shared --host=x86_64-w64-mingw32 --disable-gnu-packages \
#    --with-coinutils --with-coinutils-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-coinutils-lflags="-L/home/haroldo/prog/win64/lib/ -lCoinUtils" \
#    --with-osi --with-osi-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-osi-lflags="-L/home/haroldo/prog/win64/lib/  -lOsi"  \
#    --with-clp --with-clp-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-clp-lflags="-L/home/haroldo/prog/win64/lib/ -lClp"  \
#    --with-osiclp --with-osiclp-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-osiclp-lflags="-L/home/haroldo/prog/win64/lib/ -lOsiClp"  
#make -j 6
#make -j 6 install


cd $dir/Cbc
./configure --prefix=${IDIR} --enable-cbc-parallel --enable-static --disable-shared --host=x86_64-w64-mingw32 --disable-gnu-packages \
    --with-coinutils --with-coinutils-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-coinutils-lflags="-L/home/haroldo/prog/win64/lib/ -lCoinUtils" \
    --with-osi --with-osi-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-osi-lflags="-L/home/haroldo/prog/win64/lib/ -lOsi"  \
    --with-clp --with-clp-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-clp-lflags="-L/home/haroldo/prog/win64/lib/ -lClp"  \
    --with-osiclp --with-osiclp-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-osiclp-lflags="-L/home/haroldo/prog/win64/lib/ -lOsiClp"  \
    --with-cgl --with-cgl-cflags="-I/home/haroldo/prog/win64/include/coin-or/" --with-cgl-lflags="-L/home/haroldo/prog/win64/lib/ -lCgl"
make -j 6
make -j 6 install

exit 

cd $dir

x86_64-w64-mingw32-g++ -shared -Ofast -fPIC -o cbc-c-linux-x86-64.so \
-I${IDIR}/include/coin-or/ \
 -DCBC_THREAD \
 ./Cbc/src/Cbc_C_Interface.cpp \
  -L${IDIR}/lib/ \
 -lCbcSolver -lCbc -lpthread -lrt -lCgl -lOsiClp -lClpSolver -lClp -lOsi -lCoinUtils \
 -lcoinlapack -lcoinblas -lgfortran -lquadmath -lm -static-libgcc -static-libstdc++ -static-libgfortran -lcoinglpk
