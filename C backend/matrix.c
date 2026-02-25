#include<stdio.h>
#include<stdlib.h>

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

void Insert(Matrix * M , double key){
   if(M==NULL) return;
   
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
    if(M==NULL){
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
    if(M==NULL) return;
    
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
   if(M==NULL) return -1;

   if(M->used==0) {
       return 1;
   }else{
       return 0;
  }
}


    
Matrix * Transpose(Matrix * X){
   if(X==NULL) return NULL;

  
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
   
int main(){

  double arr[12] = { 0 , 3 , 12 , 18 , 1 , 4 , 14 , 20 , 2 , 7 , 16 , 21};
  Matrix * M = arr_to_matrix(arr , 12 , 3, 4);
 
  PrintM(M);

  printf("\nThe Last element in the array is %lf \n" , Get(M,-1,-1));
  
  printf("\nLast Row of the Matrix :\n");
  for(int i=0; i<4; i++) printf(" %lf " ,Get(M,-1,i));
  
  printf("\nLast Column of the Matrix : \n");
  for(int i=0; i < 3; i++) printf(" %lf " , Get(M, i , -1));

  Matrix * Mt = Transpose(M);
  printf("\nTranspose of M\n");
  PrintM(Mt);
  
  printf("\n Transpose of M_transpose will give back M \n");
  PrintM(Transpose(Mt));
  printf("\nAdding M and Mt \n");
  Matrix * M_add = Add(M,Mt);
  PrintM(M_add);
  
  printf("\n M + M \n");
  Matrix * M_res = Add(M,M);
  PrintM(M_res);
 
 return 0;
}    
 


