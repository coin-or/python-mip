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

