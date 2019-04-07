#include <math.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>

#include "MMult.h"

int main(int argc, char** argv)
{
    // parse input
    int nIter = 10;
    int n = 512;
    if (argc == 2)
        nIter = atoi(argv[1]);
    else if (argc == 3) {
        nIter = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    // variables
    float *A, *B, *C, *C1;
    A = new float[n * n];
    B = new float[n * n];
    C = new float[n * n];
    C1 = new float[n * n];

    // random init
    for (int i = 0; i < n * n; i++)
    {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    // timestamp
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // execute for loop
    for (int i = 0; i < nIter; i++) {
        MMult0(n, n, n, A, n, B, n, C, n);
    }
    // timestamp
    gettimeofday(&stop, NULL);
    float t0 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult0 n %d, time used %fms!\n", n, t0 / nIter);
#if 0
    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult1(n, n, n, A, n, B, n, C1, n);
	}
    gettimeofday(&stop, NULL);
    float t1 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult1 n %d, time used %fms!\n", n, t1 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult2(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t2 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult2 n %d, time used %fms!\n", n, t2 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_1x4_3(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t3 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_1x4_3 n %d, time used %fms!\n", n, t3 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_1x4_4(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t4 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_1x4_4 n %d, time used %fms!\n", n, t4 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_1x4_5(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t5 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_1x4_5 n %d, time used %fms!\n", n, t5 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_1x4_6(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t6 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_1x4_6 n %d, time used %fms!\n", n, t6 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_1x4_7(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t7 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_1x4_7 n %d, time used %fms!\n", n, t7 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_1x4_8(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t8 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_1x4_8 n %d, time used %fms!\n", n, t8 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_1x4_9(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t9 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_1x4_9 n %d, time used %fms!\n", n, t9 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_3(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t10 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_3 n %d, time used %fms!\n", n, t10 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_4(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t11 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_4 n %d, time used %fms!\n", n, t11 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_5(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t12 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_5 n %d, time used %fms!\n", n, t12 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_6(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t13 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_6 n %d, time used %fms!\n", n, t13 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_7(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t14 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_7 n %d, time used %fms!\n", n, t14 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_8(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t15 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_8 n %d, time used %fms!\n", n, t15 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_9(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t16 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_9 n %d, time used %fms!\n", n, t16 / nIter);

    // element-wise
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_10(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t17 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_10 n %d, time used %fms!\n", n, t17 / nIter);

    // element-wise
    //memset(C1, 0, n*n*sizeof(float));
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_11(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t18 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_11 n %d, time used %fms!\n", n, t18 / nIter);
#endif
    // element-wise
    //memset(C1, 0, n*n*sizeof(float));
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_12(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t19 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_12 n %d, time used %fms!\n", n, t19 / nIter);

    // element-wise
    //memset(C1, 0, n*n*sizeof(float));
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_13(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t20 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_13 n %d, time used %fms!\n", n, t20 / nIter);

    // element-wise
    //memset(C1, 0, n*n*sizeof(float));
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_14(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t21 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_14 n %d, time used %fms!\n", n, t21 / nIter);

    // element-wise
    //memset(C1, 0, n*n*sizeof(float));
    gettimeofday(&start, NULL);
    for (int i = 0; i < nIter; i++) {
        MMult_4x4_15(n, n, n, A, n, B, n, C1, n);
    }
    gettimeofday(&stop, NULL);
    float t22 = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
    printf("Done MMult_4x4_15 n %d, time used %fms!\n", n, t22 / nIter);
    
    for(int i = 0; i < n * n; i++)
    {
        float diff = fabs(C1[i] - C[i]);
        if (diff > 1e-3)
        {
            printf("ERROR: index is %d, diff is %f\n", i, diff);
            break;
        }
    }

    delete []A;
    delete []B;
    delete []C;
    delete []C1;
    return 0;
}

