#include "tensor.h"
#include <stdio.h>
#include<stdlib.h>

int main() {
    size_t shapes[] = {3,3};
    Tensor * A = createEmptyTensor(shapes , 2);
    for(int i=0; i<9; i++)
        A->data[i] = (double)i + 1;

    Tensor * A2 = TensorAdd(A,A);
    printf("\n A \n");
    printTensor(A);
    printf("A + A = 2A \n");
    printTensor(A2);
    return 0;
}
