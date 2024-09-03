#include "testfcn.h"

void increment_an_integer(int * i)
{
    *i = *i + 1;
}

void print_dmat4(double m[4][4])
{
    for(int r = 0; r < 4; r++)
    {
        for(int c = 0; c < 4; c++)
        {
            printf("%f ", m[r][c]);
        }
        printf("\n");
    }
}


