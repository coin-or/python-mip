export CFLAGS="-Og -fPIC -g3 -DDEBUG"
export FFLAGS="-Og -fPIC -g3"
export CXXFLAGS="-Og -fPIC -g3 -DDEBUG"
export LDFLAGS="-Og -fPIC -g"

dir=`pwd`
mkdir -p ~/prog
cd ~/prog
IDIR=`pwd`
export PKG_CONFIG_PATH=${IDIR}/lib/pkgconfig/:${PKG_CONFIG_PATH}

#cd $dir/ThirdParty-Glpk
#make clean ; make distclean
#./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas --enable-static --disable-shared
#make -j 6
#make -j 6 install


#cd $dir/ThirdParty-Lapack
#make clean ; make distclean
#./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas --enable-static --disable-shared 
#make -j 6
#make -j 6 install

#cd $dir/ThirdParty-Blas
#make clean ; make distclean
#./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas --enable-static --enable-shared 
#make -j 6
#make -j 6 install

cd $dir/CoinUtils
make clean ; make distclean
./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas  --enable-shared 
make -j 6
make -j 6 install

cd $dir/Osi
make clean ; make distclean
./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas  --enable-shared 
make -j 6
make -j 6 install

cd $dir/Clp
make clean ; make distclean
./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas  --enable-shared 
make -j 6
make -j 6 install

cd $dir/Cgl
make clean ; make distclean
./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas  --enable-shared 
make -j 6
make -j 6 install

cd $dir/Cbc
make clean ; make distclean
./configure --prefix=$IDIR --without-lapack --without-glpk --without-blas  --enable-shared 
make -j 6
make -j 6 install

exit


for libfile in $IDIR/lib/*.so*;
do
    chrpath -r ./ $libfile
done

chrpath -r \$\ORIGIN/../lib/ $IDIR/bin/cbc
chrpath -r \$\ORIGIN/../lib/ $IDIR/bin/clp

cd $dir
