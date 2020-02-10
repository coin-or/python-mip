echo "Checking CoinUtils"
if [[ -d CoinUtils ]]
then
    cd CoinUtils
    git pull
    cd ..
else
    svn co https://projects.coin-or.org/svn/Cbc/trunk/ Cbc
fi

echo "Checking Osi"
if [[ -d Osi ]]
then
    cd Osi
    git pull
    cd ..
else
    svn co https://projects.coin-or.org/svn/Osi/trunk/ Osi
fi

echo "Checking Clp"
if [[ -d Clp ]]
then
    cd Clp
    git pull
    cd ..
else
    svn co https://projects.coin-or.org/svn/Clp/trunk/ Clp
fi

echo "Checking Cgl"
if [[ -d Cgl ]]
then
    cd Cgl
    git pull
    cd ..
else
    svn co https://projects.coin-or.org/svn/Cgl/trunk/ Cgl
fi

echo "Checking Cbc"
if [[ -d Cbc ]]
then
    cd Cbc
    git pull
    cd ..
else
    svn co https://projects.coin-or.org/svn/Cbc/trunk/ Cbc
fi

