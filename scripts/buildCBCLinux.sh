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

export CFLAGS="-Ofast -fPIC -flto -DNDEBUG -fprefetch-loop-arrays -I/opt/gcc/include/"
export FFLAGS="-Ofast -fPIC -flto -DNDEBUG -I/opt/gcc/include/"
export CXXFLAGS="-Ofast -fPIC -flto -DNDEBUG -I/opt/gcc/include/"
export LDFLAGS="-Ofast -fPIC -L/opt/gcc/lib -flto -static-libgcc -static-libstdc++ -static-libgfortran"
./configure --prefix=~/prog/ --enable-cbc-parallel --enable-static --disable-shared 
make clean 
make -j 2
g++ -shared -Ofast -fPIC -o cbc-c-linux-x86-64.so \
-I~/prog/include/ -I./CoinUtils/src/ -I./Osi/src/ -I./Clp/src/ -I./Clp/src/ \
-I./Cgl/src/ -I./Cbc/src/ -I./Osi/src/Osi/ -I./Clp/src/OsiClp/ \
./Cbc/src/Cbc_C_Interface.cpp -L/opt/gcc/lib64/ -L./Cbc/src/.libs/ -L./Cgl/src/.libs/ \
-L./Cgl/src/.libs/ -L./Osi/src/Osi/.libs/ -L./Clp/src/OsiClp/.libs/ -L./Clp/src/.libs/ \
-L./CoinUtils/src/.libs/ -L./Cgl/src/.libs/ -L./Clp/src/OsiClp/.libs/ -L./Clp/src/.libs/ \
-L./ThirdParty/Lapack/.libs/ -L./ThirdParty/Blas/.libs/ \
 -lCbcSolver -lCbc -lpthread -lrt -lCgl -lOsiClp -lClpSolver -lClp -lOsi -lCoinUtils \
 -lcoinlapack -lcoinblas
