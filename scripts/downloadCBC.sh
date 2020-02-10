echo "Checking CoinUtils"
if [[ -d CoinUtils ]]
then
    cd CoinUtils
    git pull
    cd ..
else
    git clone https://github.com/coin-or/CoinUtils
fi

echo "Checking Osi"
if [[ -d Osi ]]
then
    cd Osi
    git pull
    cd ..
else
    git clone https://github.com/coin-or/Osi
fi

echo "Checking Clp"
if [[ -d Clp ]]
then
    cd Clp
    git pull
    cd ..
else
    git clone https://github.com/coin-or/Clp
fi

echo "Checking Cgl"
if [[ -d Cgl ]]
then
    cd Cgl
    git pull
    cd ..
else
    git clone https://github.com/coin-or/Cgl
fi

echo "Checking Cbc"
if [[ -d Cbc ]]
then
    cd Cbc
    git pull
    cd ..
else
    git clone https://github.com/coin-or/Cbc
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
