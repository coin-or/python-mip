if [ ! -d "Cbc" ];
then
	echo "call this script in the CBC source directory"
	exit
fi
if [ ! -f "configure" ];
then
	echo "call this script in the CBC source directory"
	exit
fi

export CFLAGS="-fPIC -Ofast -DNDEBUG -ffast-math"
export CXXFLAGS="-fPIC -Ofast -DNDEBUG -ffast-math"
export F77FLAGS="-fPIC -Ofast -DNDEBUG -ffast-math"
export LDFLAGS="-fPIC -Ofast -DNDEBUG -ffast-math"
./configure --prefix=/opt/cbc/bin/ --enable-static --disable-shared --without-glpk --without-gz --disable-gz --without-gz --without-z --without-zlib --without-bz2 --disable-bz2
make
make install
clang++ -Ofast -ffast-math -fPIC -shared -I/opt/cbc/bin/include/coin ./Cbc/src/Cbc_C_Interface.cpp /opt/cbc/bin/lib/libCbc.a /opt/cbc/bin/lib/libCbcSolver.a /opt/cbc/bin/lib/libOsi.a /opt/cbc/bin/lib/libClp.a /opt/cbc/bin/lib/libOsiClp.a /opt/cbc/bin/lib/libCgl.a /opt/cbc/bin/lib/libCoinUtils.a -o cbc-c-macos-x86-64.dylb -lbz2 -lz -llapack
