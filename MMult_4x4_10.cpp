#include "MMult.h"
#include <stdio.h>

/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

static void AddDot4x4( int, float *, int, float *, int, float *, int );

void MMult_4x4_10( int m, int n, int k, float *a, int lda,
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
     one routine (four inner products) */

      AddDot4x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
      //printf("i: %d, j: %d\n", i, j);
    }
  }
}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
//#include <immintrin.h>

typedef union
{
  __m128 v;
  float s[4];
} v4sf_t;

void AddDot4x4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A
           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).
     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements
           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 )
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 )
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 )
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 )

     in the original matrix C
     And now we use vector registers and instructions */

  int p;

  v4sf_t
    c00_c10_c20_c30_vreg,
    c01_c11_c21_c31_vreg,
    c02_c12_c22_c32_vreg,
    c03_c13_c23_c33_vreg,
    a0p_a1p_a2p_a3p_vreg,
    bp0_vreg, bp1_vreg, bp2_vreg, bp3_vreg;

  float
    /* Point to the current elements in the four columns of B */
    *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr;

  bp0_pntr = &B( 0, 0 );
  bp1_pntr = &B( 0, 1 );
  bp2_pntr = &B( 0, 2 );
  bp3_pntr = &B( 0, 3 );

  c00_c10_c20_c30_vreg.v = _mm_setzero_ps();
  c01_c11_c21_c31_vreg.v = _mm_setzero_ps();
  c02_c12_c22_c32_vreg.v = _mm_setzero_ps();
  c03_c13_c23_c33_vreg.v = _mm_setzero_ps();

  for ( p=0; p<k; p++ ){
    a0p_a1p_a2p_a3p_vreg.v = _mm_load_ps( (float *) &A( 0, p ) );

    bp0_vreg.v = _mm_load_ps1( (float *) bp0_pntr++ );   /* load and duplicate */
    bp1_vreg.v = _mm_load_ps1( (float *) bp1_pntr++ );   /* load and duplicate */
    bp2_vreg.v = _mm_load_ps1( (float *) bp2_pntr++ );   /* load and duplicate */
    bp3_vreg.v = _mm_load_ps1( (float *) bp3_pntr++ );   /* load and duplicate */

    /* First row and second rows */
    c00_c10_c20_c30_vreg.v += a0p_a1p_a2p_a3p_vreg.v * bp0_vreg.v;
    c01_c11_c21_c31_vreg.v += a0p_a1p_a2p_a3p_vreg.v * bp1_vreg.v;
    c02_c12_c22_c32_vreg.v += a0p_a1p_a2p_a3p_vreg.v * bp2_vreg.v;
    c03_c13_c23_c33_vreg.v += a0p_a1p_a2p_a3p_vreg.v * bp3_vreg.v;
  }

  C( 0, 0 ) = c00_c10_c20_c30_vreg.s[0];  C( 0, 1 ) = c01_c11_c21_c31_vreg.s[0];
  C( 0, 2 ) = c02_c12_c22_c32_vreg.s[0];  C( 0, 3 ) = c03_c13_c23_c33_vreg.s[0];

  C( 1, 0 ) = c00_c10_c20_c30_vreg.s[1];  C( 1, 1 ) = c01_c11_c21_c31_vreg.s[1];
  C( 1, 2 ) = c02_c12_c22_c32_vreg.s[1];  C( 1, 3 ) = c03_c13_c23_c33_vreg.s[1];

  C( 2, 0 ) = c00_c10_c20_c30_vreg.s[2];  C( 2, 1 ) = c01_c11_c21_c31_vreg.s[2];
  C( 2, 2 ) = c02_c12_c22_c32_vreg.s[2];  C( 2, 3 ) = c03_c13_c23_c33_vreg.s[2];

  C( 3, 0 ) = c00_c10_c20_c30_vreg.s[3];  C( 3, 1 ) = c01_c11_c21_c31_vreg.s[3];
  C( 3, 2 ) = c02_c12_c22_c32_vreg.s[3];  C( 3, 3 ) = c03_c13_c23_c33_vreg.s[3];
}
