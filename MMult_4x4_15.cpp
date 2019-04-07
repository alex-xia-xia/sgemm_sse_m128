#include "MMult.h"
#include <stdio.h>
#include <memory.h>

/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Block sizes */
#define mc 256
#define kc 128
#define nb 1000

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */

static void AddDot4x4( int, float *, int, float *, int, float *, int );
static void PackMatrixA( int, float *, int, float * );
static void PackMatrixB( int, float *, int, float * );
static void InnerKernel( int, int, int, float *, int, float *, int, float *, int, int );

void MMult_4x4_15( int m, int n, int k, float *a, int lda,
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, p, pb, ib;
  if (n > nb) {
    printf("n size is too large %d, nb %d\n", n, nb);
    return;
  }

  /* This time, we compute a mc x n block of C by a call to the InnerKernel */
  memset(c, 0, m*n*sizeof(float));
  for ( p=0; p<k; p+=kc ){
    pb = min( k-p, kc );
    for ( i=0; i<m; i+=mc ){
      ib = min( m-i, mc );
      InnerKernel( ib, n, pb, &A( i,p ), lda, &B(p, 0 ), ldb, &C( i,0 ), ldc, i==0 );
    }
  }
}

void InnerKernel( int m, int n, int k, float *a, int lda,
                                       float *b, int ldb,
                                       float *c, int ldc, int first_time )
{
  int i, j;
  float
    packedA[ m * k ];
  static float
    packedB[ kc*nb ];    /* Note: using a static buffer is not thread safe... */

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    if ( first_time )
      PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
     one routine (four inner products) */
      if ( j == 0 )
    PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
      AddDot4x4( k, &packedA[ i*k ], 4, &packedB[ j*k ], k, &C( i,j ), ldc );
    }
  }
}

void PackMatrixA( int k, float *a, int lda, float *a_to )
{
  int j;

  for( j=0; j<k; j++){  /* loop over columns of A */
    float
      *a_ij_pntr = &A( 0, j );

    *a_to     = *a_ij_pntr;
    *(a_to+1) = *(a_ij_pntr+1);
    *(a_to+2) = *(a_ij_pntr+2);
    *(a_to+3) = *(a_ij_pntr+3);

    a_to += 4;
  }
}

void PackMatrixB( int k, float *b, int ldb, float *b_to )
{
  int i;
  float
    *b_i0_pntr = &B( 0, 0 ), *b_i1_pntr = &B( 0, 1 ),
    *b_i2_pntr = &B( 0, 2 ), *b_i3_pntr = &B( 0, 3 );

  for( i=0; i<k; i++){  /* loop over rows of B */
    *b_to++ = *b_i0_pntr++;
    *b_to++ = *b_i1_pntr++;
    *b_to++ = *b_i2_pntr++;
    *b_to++ = *b_i3_pntr++;
  }
}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

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

  c00_c10_c20_c30_vreg.v = _mm_setzero_ps();
  c01_c11_c21_c31_vreg.v = _mm_setzero_ps();
  c02_c12_c22_c32_vreg.v = _mm_setzero_ps();
  c03_c13_c23_c33_vreg.v = _mm_setzero_ps();

  for ( p=0; p<k; p++ ){
    a0p_a1p_a2p_a3p_vreg.v = _mm_load_ps( (float *) a );
    a += 4;

    bp0_vreg.v = _mm_load_ps1( (float *) b );   /* load and duplicate */
    bp1_vreg.v = _mm_load_ps1( (float *) (b+1) );   /* load and duplicate */
    bp2_vreg.v = _mm_load_ps1( (float *) (b+2) );   /* load and duplicate */
    bp3_vreg.v = _mm_load_ps1( (float *) (b+3) );   /* load and duplicate */
    b += 4;

    /* First row and second rows */
    c00_c10_c20_c30_vreg.v += a0p_a1p_a2p_a3p_vreg.v * bp0_vreg.v;
    c01_c11_c21_c31_vreg.v += a0p_a1p_a2p_a3p_vreg.v * bp1_vreg.v;
    c02_c12_c22_c32_vreg.v += a0p_a1p_a2p_a3p_vreg.v * bp2_vreg.v;
    c03_c13_c23_c33_vreg.v += a0p_a1p_a2p_a3p_vreg.v * bp3_vreg.v;
  }

  C( 0, 0 ) += c00_c10_c20_c30_vreg.s[0];  C( 0, 1 ) += c01_c11_c21_c31_vreg.s[0];
  C( 0, 2 ) += c02_c12_c22_c32_vreg.s[0];  C( 0, 3 ) += c03_c13_c23_c33_vreg.s[0];

  C( 1, 0 ) += c00_c10_c20_c30_vreg.s[1];  C( 1, 1 ) += c01_c11_c21_c31_vreg.s[1];
  C( 1, 2 ) += c02_c12_c22_c32_vreg.s[1];  C( 1, 3 ) += c03_c13_c23_c33_vreg.s[1];

  C( 2, 0 ) += c00_c10_c20_c30_vreg.s[2];  C( 2, 1 ) += c01_c11_c21_c31_vreg.s[2];
  C( 2, 2 ) += c02_c12_c22_c32_vreg.s[2];  C( 2, 3 ) += c03_c13_c23_c33_vreg.s[2];

  C( 3, 0 ) += c00_c10_c20_c30_vreg.s[3];  C( 3, 1 ) += c01_c11_c21_c31_vreg.s[3];
  C( 3, 2 ) += c02_c12_c22_c32_vreg.s[3];  C( 3, 3 ) += c03_c13_c23_c33_vreg.s[3];
}
