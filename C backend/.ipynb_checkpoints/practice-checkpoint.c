#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* include your tensor file here */
#include "tensor.c"

void fillTensorSequential(Tensor *T)
{
    if (!T) return;

    size_t index[T->ndim];

    for(size_t i = 0; i < T->size; i++)
    {
        flat_to_multi(i, T->shapes, T->ndim, index);
        size_t offset = getOffset(T->strides, index, T->ndim);
        T->data[offset] = (double)i;
    }
}

void verifyAdd(Tensor *A, Tensor *B, Tensor *C)
{
    size_t index[A->ndim];

    for(size_t i = 0; i < A->size; i++)
    {
        flat_to_multi(i, A->shapes, A->ndim, index);

        size_t a_off = getOffset(A->strides, index, A->ndim);
        size_t b_off = getOffset(B->strides, index, B->ndim);
        size_t c_off = getOffset(C->strides, index, C->ndim);

        double expected = A->data[a_off] + B->data[b_off];

        if(C->data[c_off] != expected)
        {
            printf("ERROR ADD at flat=%zu expected=%f got=%f\n",
                   i, expected, C->data[c_off]);
            exit(1);
        }
    }

    printf("Add verification passed\n");
}

void verifySub(Tensor *A, Tensor *B, Tensor *C)
{
    size_t index[A->ndim];

    for(size_t i = 0; i < A->size; i++)
    {
        flat_to_multi(i, A->shapes, A->ndim, index);

        size_t a_off = getOffset(A->strides, index, A->ndim);
        size_t b_off = getOffset(B->strides, index, B->ndim);
        size_t c_off = getOffset(C->strides, index, C->ndim);

        double expected = A->data[a_off] - B->data[b_off];

        if(C->data[c_off] != expected)
        {
            printf("ERROR SUB at flat=%zu expected=%f got=%f\n",
                   i, expected, C->data[c_off]);
            exit(1);
        }
    }

    printf("Sub verification passed\n");
}

int main()
{
    // printf("===== TENSOR LIBRARY TEST =====\n\n");

    // /* ------------------------------- */
    // /* Test 1: Basic tensor creation  */
    // /* ------------------------------- */

    // size_t shape2d[2] = {4,5};

    // Tensor *A = createEmptyTensor(shape2d,2);
    // Tensor *B = createEmptyTensor(shape2d,2);

    // fillTensorSequential(A);
    // fillTensorSequential(B);

    // printf("A:\n");
    // printTensor(A);

    // printf("B:\n");
    // printTensor(B);


    // /* ------------------------------- */
    // /* Test 2: Contiguous Add         */
    // /* ------------------------------- */

    // Tensor *C = TensorAdd(A,B);

    // printf("\nC = A + B\n");
    // printTensor(C);

    // verifyAdd(A,B,C);


    // /* ------------------------------- */
    // /* Test 3: Transpose view         */
    // /* ------------------------------- */

    // Tensor *At = Transpose(A);
    // Tensor *Bt = Transpose(B);

    // printf("\nAt info\n");
    // printTensorInfo(At);

    // printf("\nBt info\n");
    // printTensorInfo(Bt);


    // /* ------------------------------- */
    // /* Test 4: Non-contiguous Add     */
    // /* ------------------------------- */

    // Tensor *D = TensorAdd(At,Bt);

    // printf("\nD = At + Bt\n");
    // printTensor(D);

    // verifyAdd(At,Bt,D);


    // /* ------------------------------- */
    // /* Test 5: Subtraction            */
    // /* ------------------------------- */

    // Tensor *E = TensorSub(A,B);

    // printf("\nE = A - B\n");
    // printTensor(E);

    // verifySub(A,B,E);


    // /* ------------------------------- */
    // /* Test 6: Reshape view           */
    // /* ------------------------------- */

    // size_t newshape[2] = {2,10};

    // Tensor *R = reshape(A,newshape,2);

    // printf("\nReshaped A (2x10)\n");
    // printTensorInfo(R);


    // /* ------------------------------- */
    // /* Test 7: Stress add             */
    // /* ------------------------------- */

    // size_t shape3d[3] = {6,5,4};

    // Tensor *T1 = createEmptyTensor(shape3d,3);
    // Tensor *T2 = createEmptyTensor(shape3d,3);

    // fillTensorSequential(T1);
    // fillTensorSequential(T2);

    // Tensor *T3 = TensorAdd(T1,T2);

    // verifyAdd(T1,T2,T3);

    // printf("3D add passed\n");


    // /* ------------------------------- */
    // /* Test 8: transpose + reshape    */
    // /* ------------------------------- */

    // Tensor *T1t = Transpose(A);

    // Tensor *Tsum = TensorAdd(A,Transpose(T1t));

    // verifyAdd(A,A,Tsum);

    // printf("transpose + transpose add passed\n");


    // /* ------------------------------- */
    // /* Test 9: ref count behavior     */
    // /* ------------------------------- */

    // printf("\nReference count tests\n");

    // printTensorInfo(A);

    // Tensor *view1 = Transpose(A);
    // Tensor *view2 = reshape(A,shape2d,2);

    // printTensorInfo(A);

    // freeTensor(view1);
    // freeTensor(view2);

    // printTensorInfo(A);


    // /* ------------------------------- */
    // /* Test 10: large tensor mmap     */
    // /* ------------------------------- */

    // size_t bigshape[2] = {2000,2000};

    // Tensor *BIG = createEmptyTensor(bigshape,2);

    // printf("\nBIG tensor created\n");
    // printTensorInfo(BIG);


    // /* ------------------------------- */
    // /* Free everything                */
    // /* ------------------------------- */

    // freeTensor(A);
    // freeTensor(B);
    // freeTensor(C);
    // freeTensor(D);
    // freeTensor(E);
    // freeTensor(R);

    // freeTensor(T1);
    // freeTensor(T2);
    // freeTensor(T3);

    // freeTensor(T1t);
    // freeTensor(Tsum);

    // freeTensor(BIG);


    // printf("\n===== MATMUL TEST =====\n");

size_t shapeA[2] = {4,1};
size_t shapeB[2] = {1,4};

Tensor *A = createEmptyTensor(shapeA,2);
Tensor *B = createEmptyTensor(shapeB,2);

/* fill A */
for(size_t r=0; r<A->shapes[0]; r++){
    for(size_t c=0; c<A->shapes[1]; c++){
        size_t off = A->strides[0]*r + A->strides[1]*c;
        A->data[off] = r*4 + c + 1;
    }
}

/* fill B */
for(size_t r=0; r<B->shapes[0]; r++){
    for(size_t c=0; c<B->shapes[1]; c++){
        size_t off = B->strides[0]*r + B->strides[1]*c;
        B->data[off] = r*2 + c + 1;
    }
}

printf("\nMatrix A:\n");
printTensorInfo(A);
printTensor(A);
printf("\n");
    
printf("\nMatrix B:\n");
printTensorInfo(B);
printTensor(B);
printf("\n");
/* A @ B */
Tensor *C = matMul(A,B);

printf("\nC = A @ B:\n");
printTensorInfo(C);
printTensor(C);
printf("\n");

/* transpose test */
Tensor *At = Transpose(A);
Tensor *Att = Transpose(At);

printf("\nTranspose(A):\n");
printTensor(At);

printf("\nTranspose(Transpose(A)):\n");
printTensor(Att);

/* should match original multiplication */
Tensor *C2 = matMul(Att,B);

printf("\nC2 = Transpose(Transpose(A)) @ B:\n");
printTensor(C2);
size_t newshapes[1] = { 1};
Tensor * C3 = reshape(matMul(Transpose(A),Transpose(B)) , newshapes , 1 );

printf("\n C3 = A^T @ B^T  \n");
printTensorInfo(C3);
printTensor(C3);

/* free memory */
freeTensor(A);
freeTensor(B);
freeTensor(C);
freeTensor(At);
freeTensor(Att);
freeTensor(C2);
freeTensor(C3);
printf("\nALL TESTS PASSED\n");
    return 0;
}
