#include "MMult.h"

/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

static void AddDot( int, float *, int, float *, float * );

void MMult2( int m, int n, int k, float *a, int lda,
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    for ( i=0; i<m; i+=1 ){        /* Loop over the rows of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
     and the jth column of B */

      AddDot( k, &A( i,0 ), lda, &B( 0,j ), &C( i,j ) );

      /* Update the C( i,j+1 ) with the inner product of the ith row of A
     and the (j+1)th column of B */

      AddDot( k, &A( i,0 ), lda, &B( 0,j+1 ), &C( i,j+1 ) );

      /* Update the C( i,j+2 ) with the inner product of the ith row of A
     and the (j+2)th column of B */

      AddDot( k, &A( i,0 ), lda, &B( 0,j+2 ), &C( i,j+2 ) );

      /* Update the C( i,j+3 ) with the inner product of the ith row of A
     and the (j+1)th column of B */

      AddDot( k, &A( i,0 ), lda, &B( 0,j+3 ), &C( i,j+3 ) );
    }
  }
}


/* Create macro to let X( i ) equal the ith element of x */

#define X(i) x[ (i)*incx ]

static void AddDot( int k, float *x, int incx,  float *y, float *gamma )
{
  /* compute gamma := x' * y + gamma with vectors x and y of length n.

     Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
  */

  int p;

  *gamma = 0;
  for ( p=0; p<k; p++ ){
    *gamma += X( p ) * y[ p ];
  }
}
