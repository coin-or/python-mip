#include "/home/haroldo/src/cbctrunk/Cbc/src/Cbc_C_Interface.h"

int main(int argc, char **argv)
{
    Cbc_Model *m = Cbc_newModel();
    Cbc_readMps(m, "/home/haroldo/air04.mps");
    Cbc_solve(m);
    Cbc_deleteModel(m);
}

