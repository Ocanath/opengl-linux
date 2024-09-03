#include "testfcn.h"

int main(void)
{
    int i = 0; 
    increment_an_integer(&i);
    printf("%d\r\n", i);

    double mat4[4][4] = {0};
    for(int i = 0; i < 4; i++)
        mat4[i][i] = 1.;
    print_dmat4(mat4);
}
