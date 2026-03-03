#include<stdio.h>
#include<stdlib.h>
#include <unistd.h>

typedef struct Matrix{
  int rows;
  int cols;
  double * arr; 
  int used;    
}Matrix;


Matrix * newEmptyMatrix(int rows , int cols){
     if(rows < 0) rows = -1 * rows;
     if(cols < 0) cols = -1 * cols;

     double * Matrix_arr = (double *)calloc((rows*cols),sizeof(double));
     Matrix * M = (Matrix *)malloc(sizeof(Matrix));
     M->rows = rows;
     M->cols = cols;
     M->arr = Matrix_arr;
     M->used =0;
     return M;
}

int getSize(Matrix * M ){
   if(M==NULL) return -1;
   if(M->rows == 0 || M->cols ==0 ){
        return 0;
   }

   return (M->rows * M->cols);
}
int isNull(Matrix * M){
    if(M==NULL){
       return 1;
    }else{
       return 0;
    }
}
void Insert(Matrix * M , double key){
   if(isNull(M)) return;
   
   if(M->used == getSize(M)) {
        printf("Error : size is full ");
        exit(1);
   }
   int used = M->used;
   
   M->arr[used] = key;
   
   M->used++;
}
int hash(int i, int j,int col_size){
    int k = (i * col_size) + j;
    return k;
}


double Get(Matrix * M , int row_index , int col_index ){
    if(isNull(M)){
        printf("NULL pointer is invalid\n");
        exit(1);
    }

    if(row_index >= M->rows || col_index >=M->cols){
        printf("In Valid Index \n");
        exit(1);
    }
    
    if(row_index == -1){
       row_index = M->rows - 1;
    }
    
    if(col_index == -1){
        col_index = M->cols -1;
    }
     
    if(row_index < -1 || col_index < -1){
            printf("Index cannot be < -1 \n");
            exit(1);
    }
        
    int index = hash(row_index , col_index , M->cols);

    return M->arr[index];
}       


void PrintM(Matrix * M){
    if(isNull(M)) return;
    
    int rows = M->rows;
    int cols = M->cols;
    if(M->used ==0){
        printf("Matrix is Empty\n");
        exit(1);
    }
    for(int i=0; i<rows; i++){
        for( int j=0; j<cols; j++){
              printf(" %lf ",Get(M,i,j));
        }
        printf("\n");
     }
    
}  

Matrix * arr_to_matrix(double * arr , int n , int row , int col){
  if(arr==NULL) return NULL;

  if(n > (row*col) ){
       printf("ERROR: the total size of matrix is %d and total size array given %d \n",(row*col) , n);
       exit(1);
  }
  Matrix * M = newEmptyMatrix(row,col);  
  

  for(int i=0; i<n; i++) Insert(M,arr[i]);
  return M;

}

int isEmpty(Matrix * M){
   if(isNull(M)) return -1;

   if(M->used==0) {
       return 1;
   }else{
       return 0;
  }
}


    
Matrix * Transpose(Matrix * X){
   if(isNull(X)) return NULL;

  
   if(isEmpty(X)){
       printf("The Matrix is Empty returning an empty matrix \n");
       return newEmptyMatrix(X->cols , X->rows);
   }
  
   
   
   Matrix * Xt = newEmptyMatrix(X->cols , X->rows); 
   for (int c=0; c<X->cols; c++){
       for(int r=0; r<X->rows; r++){
            Insert(Xt,Get(X ,r,c));
       }
   }   
   
   return Xt;
}

Matrix * Add(Matrix * M1 , Matrix * M2){
     // Add matrix element by element
     if(isNull(M1) || isNull(M2)){
         printf("Invalid null pointer error \n");
         exit(1);
     }
   
     if(M1->rows == M2->rows && M1->cols == M2->cols){
          // Only then we can add 
          int R = M1->rows;
          int C = M1->cols;
          double res;
          Matrix * Res = newEmptyMatrix(R,C);
          for(int r =0; r <R; r++){
             for(int c=0; c<C; c++){
                     res = Get(M1,r,c) + Get(M2,r,c);
                     Insert(Res , res);
             }
          }
          return Res;
     }else{
           printf("ERROR: invalid size of Matrix op1(%d,%d) op2(%d,%d) should be same \n",M1->rows , M1->cols , M2->rows , M2->cols);
           return NULL;
           exit(1);
     }
}

Matrix * scalling_matrix(Matrix * M , double alpha ){
   if(isNull(M)) return NULL;

   int R = M->rows;
   int C = M->cols;
   Matrix * M_new = newEmptyMatrix(R,C);
   for (int r=0; r<R; r++){
      for(int c=0; c<C; c++){
            double value = Get(M,r,c) * alpha;
            Insert(M_new , value);
      }
   }
   return M_new;

}

 
void reshape(Matrix * M , int rows , int cols){
    if(isNull(M)){
        printf("\n Null Pointer ERROR \n");
        exit(1);
    }        
    int size = M->rows * M->cols;
    int new_size = rows * cols;
    if(size == new_size){
         M->rows = rows;
         M->cols = cols;
    }else if(size < new_size){
     // we need to perform realloc here to increase the size
        double * temp = realloc(M->arr , new_size * sizeof(double));
        if(!temp){
            perror("realloc failed");
            exit(1);
        }
        M->arr = temp;
        for(int i = size; i < new_size; i++)
              M->arr[i]=0; // Zero initialize all the newly created memeory
        M->rows = rows;
        M->cols = cols;
    }else{
         printf("ERROR: the given size is %d which is < %d ", new_size , size);
         exit(1);
    }
}

Matrix * matMul(Matrix * A , Matrix * B){
   if(isNull(A) || isNull(B)){
        printf("ERROR: invalid pointer \n");
        exit(1);
   }
   if(A->cols == B->rows){
      // the inner dimension should match 
      int Row_A = A->rows;
      int Col_B = B->cols;
      int Inner = A->cols; // or B->rows;
      double C = 0;
      Matrix * res = newEmptyMatrix(Row_A , Col_B); 

      for(int r=0; r<Row_A; r++){
           for(int c=0; c<Col_B; c++){
               for(int k=0; k<Inner; k++){
                   C = C + Get(A,r,k) * Get(B,k,c);
               }
               Insert(res , C);
               C=0; 
           }
      }
      
      return res;

    }else{
        printf("\nERROR: the dimensions should match (%d , %d) ? (%d , %d) \n", A->rows , A->cols , B->rows , B->cols);
        exit(1);
    }
 
};

int main(){

  double arr[12] = { 0 , 3 , 12 , 18 , 1 , 4 , 14 , 20 , 2 , 7 , 16 , 21};
  Matrix * A = arr_to_matrix(arr , 12 , 3, 4);
  printf("A = \n"); PrintM(A);
  
  Matrix * B = Transpose(A);
  printf("B = \n"); PrintM(B);  
  printf("\n Performing matrix multiplication AA^T or AtA all results a square matrix \n");
  Matrix * C = matMul(A,B);
  
  PrintM(C);
  
  double arr2[12] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 };
  Matrix * A2 = arr_to_matrix(arr , 12 , 3 , 4);
  printf("A2 =\n");
  PrintM(A2);
  printf("A^t * ( A + A2)\n");
  PrintM(matMul(Transpose(A) , Add(A , A2)));

 return 0;
}    
 


