#include "MMult.h"

/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void MMult0( int m, int n, int k, float *a, int lda,
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j, p;

  for ( j=0; j<n; j++ ) {        /* Loop over the columns of C */
    for ( i=0; i<m; i++ ) {        /* Loop over the rows of C */
      float sum = 0.f;
      for ( p=0; p<k; p++ ) {        /* Update C( i,j ) with the inner
                       product of the ith row of A and
                       the jth column of B */
        sum +=  A( i,p ) * B( p,j );
      }
      C(i, j) = sum;
    }
  }
}
