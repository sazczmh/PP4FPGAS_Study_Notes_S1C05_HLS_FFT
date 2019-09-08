#ifndef FFT_H_
#define FFT_H_

typedef float DTYPE;
typedef int INTTYPE;
#define M 10 			/* Number of Stages = Log2N */
#define SIZE 1024 		/* SIZE OF FFT */
#define SIZE2 SIZE>>1	/* SIZE/2 */

//#define S1_baseline
//#define S2_Code_Restructured
//#define S3_LUT
//#define S4_DATAFLOW
#define S5_Effect_Improve


#ifdef S1_baseline
void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE]);
#endif


#ifdef S2_Code_Restructured
void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]);
#endif

#ifdef S3_LUT
void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]);
#endif

#ifdef S4_DATAFLOW
void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]);
#endif

#ifdef S5_Effect_Improve
void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]);
#endif

//W_real and W_image are twiddle factors for 1024 size FFT.
//WW_R[i]=cos(e*i/SIZE);
//WW_I[i]=sin(e*i/SIZE);
//where i=[0,512) and DTYPE	e = -6.283185307178;
#include "tw_r.h"
#include "tw_i.h"
#include "coefficients1024.h"

#endif
