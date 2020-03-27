echo "Checking CoinUtils"
if [[ -d CoinUtils ]]
then
    cd CoinUtils
    svn up
    cd ..
else
    svn co https://projects.coin-or.org/svn/CoinUtils/trunk/ CoinUtils
fi

echo "Checking Osi"
if [[ -d Osi ]]
then
    cd Osi
    svn up
    cd ..
else
    svn co https://projects.coin-or.org/svn/Osi/trunk/ Osi
fi

echo "Checking Clp"
if [[ -d Clp ]]
then
    cd Clp
    svn up
    cd ..
else
    svn co https://projects.coin-or.org/svn/Clp/trunk/ Clp
fi

echo "Checking Cgl"
if [[ -d Cgl ]]
then
    cd Cgl
    svn up
    cd ..
else
    svn co https://projects.coin-or.org/svn/Cgl/trunk/ Cgl
fi

echo "Checking Cbc"
if [[ -d Cbc ]]
then
    cd Cbc
    svn up
    cd ..
else
    svn co https://projects.coin-or.org/svn/Cbc/trunk/ Cbc
fi

echo "Checking Glpk"
if [[ -d ThirdParty-Glpk ]]
then
    cd ThirdParty-Glpk
    git pull
    ./get.Glpk
    cd ..
else
    git clone https://github.com/coin-or-tools/ThirdParty-Glpk.git
    cd ThirdParty-Glpk
    ./get.Glpk
    cd ../
fi

echo "Checking Lapack"
if [[ -d ThirdParty-Lapack ]]
then
    cd ThirdParty-Lapack
    git pull
    ./get.Lapack
    cd ..
else
    git clone https://github.com/coin-or-tools/ThirdParty-Lapack.git
    cd ThirdParty-Lapack
    ./get.Lapack
    cd ../
fi

echo "Checking Blas"
if [[ -d ThirdParty-Blas ]]
then
    cd ThirdParty-Blas
    git pull
    ./get.Blas
    cd ..
else
    git clone https://github.com/coin-or-tools/ThirdParty-Blas.git
    cd ThirdParty-Blas
    ./get.Blas
    cd ../
fi

echo "Checking Mumps"
if [[ -d ThirdParty-Mumps ]]
then
    cd ThirdParty-Mumps
    git pull
    ./get.Mumps
    cd ..
else
    git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
    cd ThirdParty-Mumps
    ./get.Mumps
    cd ../
fi

echo "Checking Metis"
if [[ -d ThirdParty-Metis ]]
then
    cd ThirdParty-Metis
    git pull
    ./get.Metis
    cd ..
else
    git clone https://github.com/coin-or-tools/ThirdParty-Metis.git
    cd ThirdParty-Metis
    ./get.Metis
    cd ../
fi



