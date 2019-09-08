#include "math.h"
#include "fft.h"

#ifdef S1_baseline


unsigned int reverse_bits(unsigned int input) {
	int i, rev = 0;
	for (i = 0; i < M; i++) {
		rev = (rev << 1) | (input & 1);
		input = input >> 1;
	}
	return rev;
}

void bit_reverse(DTYPE X_R[SIZE], DTYPE X_I[SIZE]) {
	unsigned int reversed;
	unsigned int i;
	DTYPE temp;

	for (i = 0; i < SIZE; i++) {
		reversed = reverse_bits(i); // Find the bit reversed index
		if (i <= reversed) {
			// Swap the real values
			temp = X_R[i];
			X_R[i] = X_R[reversed];
			X_R[reversed] = temp;

			// Swap the imaginary values
			temp = X_I[i];
			X_I[i] = X_I[reversed];
			X_I[reversed] = temp;
		}
	}
}

void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE]) {
	DTYPE temp_R; // temporary storage complex variable
	DTYPE temp_I; // temporary storage complex variable
	int i, j, k;	// loop indexes
	int i_lower;	// Index of lower point in butterfly
	int step, stage, DFTpts;
	int numBF;			// Butterfly Width
	int N2 = SIZE2; // N2=N>>1

	bit_reverse(X_R, X_I);

	step = N2;
	DTYPE a, e, c, s;

stage_loop:
	for (stage = 1; stage <= M; stage++) { // Do M stages of butterflies
		DFTpts = 1 << stage;								 // DFT = 2^stage = points in sub DFT
		numBF = DFTpts / 2;									 // Butterfly WIDTHS in sub-DFT
		k = 0;
		e = -6.283185307178 / DFTpts;
		a = 0.0;
	// Perform butterflies for j-th stage
	butterfly_loop:
		for (j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
			c = cos(a);
			s = sin(a);
			a = a + e;
		// Compute butterflies that use same W**k
		dft_loop:
			for (i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
				i_lower = i + numBF; // index of lower point in butterfly
				temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
				temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
				X_R[i_lower] = X_R[i] - temp_R;
				X_I[i_lower] = X_I[i] - temp_I;
				X_R[i] = X_R[i] + temp_R;
				X_I[i] = X_I[i] + temp_I;
			}
			k += step;
		}
		step = step / 2;
	}
}
#endif


#ifdef S2_Code_Restructured

unsigned int reverse_bits(unsigned int input) {
	int i, rev = 0;
	for (i = 0; i < M; i++) {
		rev = (rev << 1) | (input & 1);
		input = input >> 1;
	}
	return rev;
}

void bit_reverse(DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		 DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]) {
  unsigned int reversed;
  unsigned int i;


  for (int i = 0; i < SIZE; i++) {
//#pragma HLS PIPELINE
	  reversed = reverse_bits(i); // Find the bit reversed index
		if (i <= reversed) {
			// Swap the real values
			OUT_R[i] = X_R[reversed];
			OUT_R[reversed] = X_R[i];

			// Swap the imaginary values
			OUT_I[i] = X_I[reversed];
			OUT_I[reversed] = X_I[i];
		}
	}
}
void fft_stage_1( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int DFTpts = 1 << 1;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> 1;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	DTYPE a = 0.0;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1
		DTYPE c = cos(a);
		DTYPE s = sin(a);
		a = a + e;
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_2( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int DFTpts = 1 << 2;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> 2;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	DTYPE a = 0.0;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2
		DTYPE c = cos(a);
		DTYPE s = sin(a);
		a = a + e;
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_3( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int DFTpts = 1 << 3;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> 3;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	DTYPE a = 0.0;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
		DTYPE c = cos(a);
		DTYPE s = sin(a);
		a = a + e;
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_4( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int DFTpts = 1 << 4;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> 4;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	DTYPE a = 0.0;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
		DTYPE c = cos(a);
		DTYPE s = sin(a);
		a = a + e;
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_5( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int DFTpts = 1 << 5;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> 5;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	DTYPE a = 0.0;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
		DTYPE c = cos(a);
		DTYPE s = sin(a);
		a = a + e;
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_6( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int DFTpts = 1 << 6;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> 6;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	DTYPE a = 0.0;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32
		DTYPE c = cos(a);
		DTYPE s = sin(a);
		a = a + e;
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_7( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int DFTpts = 1 << 7;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> 7;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	DTYPE a = 0.0;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
		DTYPE c = cos(a);
		DTYPE s = sin(a);
		a = a + e;
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_8( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int DFTpts = 1 << 8;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> 8;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	DTYPE a = 0.0;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
		DTYPE c = cos(a);
		DTYPE s = sin(a);
		a = a + e;
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_9( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int DFTpts = 1 << 9;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> 9;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	DTYPE a = 0.0;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
		DTYPE c = cos(a);
		DTYPE s = sin(a);
		a = a + e;
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_10( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int DFTpts = 1 << 10;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> 10;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	DTYPE a = 0.0;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
		DTYPE c = cos(a);
		DTYPE s = sin(a);
		a = a + e;
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}


void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]) {



	DTYPE	Stage1_R[SIZE], Stage1_I[SIZE];
	DTYPE	Stage2_R[SIZE], Stage2_I[SIZE];
	DTYPE	Stage3_R[SIZE], Stage3_I[SIZE];
	DTYPE	Stage4_R[SIZE], Stage4_I[SIZE];
	DTYPE	Stage5_R[SIZE], Stage5_I[SIZE];
	DTYPE	Stage6_R[SIZE], Stage6_I[SIZE];
	DTYPE	Stage7_R[SIZE], Stage7_I[SIZE];
	DTYPE	Stage8_R[SIZE], Stage8_I[SIZE];
	DTYPE	Stage9_R[SIZE], Stage9_I[SIZE];
	DTYPE	Stage10_R[SIZE], Stage10_I[SIZE];

	bit_reverse(X_R, X_I, Stage1_R, Stage1_I);
	fft_stage_1(Stage1_R, Stage1_I, Stage2_R, Stage2_I);
	fft_stage_2(Stage2_R, Stage2_I, Stage3_R, Stage3_I);
	fft_stage_3(Stage3_R, Stage3_I, Stage4_R, Stage4_I);
	fft_stage_4(Stage4_R, Stage4_I, Stage5_R, Stage5_I);
	fft_stage_5(Stage5_R, Stage5_I, Stage6_R, Stage6_I);
	fft_stage_6(Stage6_R, Stage6_I, Stage7_R, Stage7_I);
	fft_stage_7(Stage7_R, Stage7_I, Stage8_R, Stage8_I);
	fft_stage_8(Stage8_R, Stage8_I, Stage9_R, Stage9_I);
	fft_stage_9(Stage9_R, Stage9_I, Stage10_R, Stage10_I);
	fft_stage_10(Stage10_R, Stage10_I, OUT_R, OUT_I);
}

#endif

#ifdef S3_LUT

unsigned int reverse_bits(unsigned int input) {
	int i, rev = 0;
	for (i = 0; i < M; i++) {
		rev = (rev << 1) | (input & 1);
		input = input >> 1;
	}
	return rev;
}

void bit_reverse(DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		 DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]) {
  unsigned int reversed;
  unsigned int i;


  for (int i = 0; i < SIZE; i++) {
//#pragma HLS PIPELINE
	  reversed = reverse_bits(i); // Find the bit reversed index
		if (i <= reversed) {
			// Swap the real values
			OUT_R[i] = X_R[reversed];
			OUT_R[reversed] = X_R[i];

			// Swap the imaginary values
			OUT_I[i] = X_I[reversed];
			OUT_I[reversed] = X_I[i];
		}
	}
}
void fft_stage_1( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 1;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> stage;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_2( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 2;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> stage;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_3( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 3;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> stage;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_4( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 4;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> stage;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_5( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 5;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> stage;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_6( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 6;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> stage;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_7( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 7;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> stage;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_8( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 8;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> stage;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_9( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 9;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> stage;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}
void fft_stage_10( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 10;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	int step = SIZE >> stage;
	DTYPE k = 0;
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		k += step;
		}
}


void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]) {



	DTYPE	Stage1_R[SIZE], Stage1_I[SIZE];
	DTYPE	Stage2_R[SIZE], Stage2_I[SIZE];
	DTYPE	Stage3_R[SIZE], Stage3_I[SIZE];
	DTYPE	Stage4_R[SIZE], Stage4_I[SIZE];
	DTYPE	Stage5_R[SIZE], Stage5_I[SIZE];
	DTYPE	Stage6_R[SIZE], Stage6_I[SIZE];
	DTYPE	Stage7_R[SIZE], Stage7_I[SIZE];
	DTYPE	Stage8_R[SIZE], Stage8_I[SIZE];
	DTYPE	Stage9_R[SIZE], Stage9_I[SIZE];
	DTYPE	Stage10_R[SIZE], Stage10_I[SIZE];

	bit_reverse(X_R, X_I, Stage1_R, Stage1_I);
	fft_stage_1(Stage1_R, Stage1_I, Stage2_R, Stage2_I);
	fft_stage_2(Stage2_R, Stage2_I, Stage3_R, Stage3_I);
	fft_stage_3(Stage3_R, Stage3_I, Stage4_R, Stage4_I);
	fft_stage_4(Stage4_R, Stage4_I, Stage5_R, Stage5_I);
	fft_stage_5(Stage5_R, Stage5_I, Stage6_R, Stage6_I);
	fft_stage_6(Stage6_R, Stage6_I, Stage7_R, Stage7_I);
	fft_stage_7(Stage7_R, Stage7_I, Stage8_R, Stage8_I);
	fft_stage_8(Stage8_R, Stage8_I, Stage9_R, Stage9_I);
	fft_stage_9(Stage9_R, Stage9_I, Stage10_R, Stage10_I);
	fft_stage_10(Stage10_R, Stage10_I, OUT_R, OUT_I);
}

#endif


#ifdef S4_DATAFLOW

unsigned int reverse_bits(unsigned int input) {
	int i, rev = 0;
	for (i = 0; i < M; i++) {
		rev = (rev << 1) | (input & 1);
		input = input >> 1;
	}
	return rev;
}

void bit_reverse(DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		 DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]) {
  unsigned int reversed;
  unsigned int i;


  for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE II=2
	  reversed = reverse_bits(i); // Find the bit reversed index
		if (i <= reversed) {
			// Swap the real values
			OUT_R[i] = X_R[reversed];
			OUT_R[reversed] = X_R[i];

			// Swap the imaginary values
			OUT_I[i] = X_I[reversed];
			OUT_I[reversed] = X_I[i];
		}
	}
}
void fft_stage_1( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 1;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {

#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }

		}
}
void fft_stage_2( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 2;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_3( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 3;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_4( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 4;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_5( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 5;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_6( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 6;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_7( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 7;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_8( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 8;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT

	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_9( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 9;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_10( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 10;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}


void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]) {
#pragma HLS DATAFLOW



	DTYPE	Stage1_R[SIZE], Stage1_I[SIZE];
	DTYPE	Stage2_R[SIZE], Stage2_I[SIZE];
	DTYPE	Stage3_R[SIZE], Stage3_I[SIZE];
	DTYPE	Stage4_R[SIZE], Stage4_I[SIZE];
	DTYPE	Stage5_R[SIZE], Stage5_I[SIZE];
	DTYPE	Stage6_R[SIZE], Stage6_I[SIZE];
	DTYPE	Stage7_R[SIZE], Stage7_I[SIZE];
	DTYPE	Stage8_R[SIZE], Stage8_I[SIZE];
	DTYPE	Stage9_R[SIZE], Stage9_I[SIZE];
	DTYPE	Stage10_R[SIZE], Stage10_I[SIZE];

	bit_reverse(X_R, X_I, Stage1_R, Stage1_I);
	fft_stage_1(Stage1_R, Stage1_I, Stage2_R, Stage2_I);
	fft_stage_2(Stage2_R, Stage2_I, Stage3_R, Stage3_I);
	fft_stage_3(Stage3_R, Stage3_I, Stage4_R, Stage4_I);
	fft_stage_4(Stage4_R, Stage4_I, Stage5_R, Stage5_I);
	fft_stage_5(Stage5_R, Stage5_I, Stage6_R, Stage6_I);
	fft_stage_6(Stage6_R, Stage6_I, Stage7_R, Stage7_I);
	fft_stage_7(Stage7_R, Stage7_I, Stage8_R, Stage8_I);
	fft_stage_8(Stage8_R, Stage8_I, Stage9_R, Stage9_I);
	fft_stage_9(Stage9_R, Stage9_I, Stage10_R, Stage10_I);
	fft_stage_10(Stage10_R, Stage10_I, OUT_R, OUT_I);
}

#endif


#ifdef S5_Effect_Improve

unsigned int reverse_bits(unsigned int input) {
	int i, rev = 0;
	for (i = 0; i < M; i++) {
		rev = (rev << 1) | (input & 1);
		input = input >> 1;
	}
	return rev;
}

void bit_reverse(DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		 DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]) {
  unsigned int reversed;
  unsigned int i;


  for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE II=2
	  reversed = reverse_bits(i); // Find the bit reversed index
		if (i <= reversed) {
			// Swap the real values
			OUT_R[i] = X_R[reversed];
			OUT_R[reversed] = X_R[i];

			// Swap the imaginary values
			OUT_I[i] = X_I[reversed];
			OUT_I[reversed] = X_I[i];
		}
	}
}
void fft_stage_1( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 1;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {

#pragma HLS PIPELINE II=6
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }

		}
}
void fft_stage_2( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 2;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=6
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_3( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 3;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=6
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_4( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 4;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=4
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_5( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 5;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=4
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_6( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 6;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=4
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_7( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 7;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=4
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_8( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 8;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT

	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_9( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 9;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
#pragma HLS PIPELINE II=10
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		int i = j;
		int i_lower = i + numBF; // index of lower point in butterfly
		int i1 = i + DFTpts;
		int i1_lower = i1 + numBF; // index of lower point in butterfly

		DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;

		Out_R[i_lower] = X_R[i] - temp_R;
		Out_I[i_lower] = X_I[i] - temp_I;
		Out_R[i] = X_R[i] + temp_R;
		Out_I[i] = X_I[i] + temp_I;

		DTYPE temp1_R = X_R[i1_lower] * c - X_I[i1_lower] * s;
		DTYPE temp1_I = X_I[i1_lower] * c + X_R[i1_lower] * s;
		Out_R[i1_lower] = X_R[i1] - temp1_R;
		Out_I[i1_lower] = X_I[i1] - temp1_I;
		Out_R[i1] = X_R[i1] + temp1_R;
		Out_I[i1] = X_I[i1] + temp1_I;
		}

}
void fft_stage_10( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 10;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
		// Compute butterflies that use same W**k
		dft_loop:
#pragma HLS PIPELINE II=6

			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = j + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[j] - temp_R;
		  Out_I[i_lower] = X_I[j] - temp_I;
		  Out_R[j] = X_R[j] + temp_R;
		  Out_I[j] = X_I[j] + temp_I;

	}
}


void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]) {
#pragma HLS DATAFLOW



	DTYPE	Stage1_R[SIZE], Stage1_I[SIZE];
	DTYPE	Stage2_R[SIZE], Stage2_I[SIZE];
	DTYPE	Stage3_R[SIZE], Stage3_I[SIZE];
	DTYPE	Stage4_R[SIZE], Stage4_I[SIZE];
	DTYPE	Stage5_R[SIZE], Stage5_I[SIZE];
	DTYPE	Stage6_R[SIZE], Stage6_I[SIZE];
	DTYPE	Stage7_R[SIZE], Stage7_I[SIZE];
	DTYPE	Stage8_R[SIZE], Stage8_I[SIZE];
	DTYPE	Stage9_R[SIZE], Stage9_I[SIZE];
	DTYPE	Stage10_R[SIZE], Stage10_I[SIZE];

	bit_reverse(X_R, X_I, Stage1_R, Stage1_I);
	fft_stage_1(Stage1_R, Stage1_I, Stage2_R, Stage2_I);
	fft_stage_2(Stage2_R, Stage2_I, Stage3_R, Stage3_I);
	fft_stage_3(Stage3_R, Stage3_I, Stage4_R, Stage4_I);
	fft_stage_4(Stage4_R, Stage4_I, Stage5_R, Stage5_I);
	fft_stage_5(Stage5_R, Stage5_I, Stage6_R, Stage6_I);
	fft_stage_6(Stage6_R, Stage6_I, Stage7_R, Stage7_I);
	fft_stage_7(Stage7_R, Stage7_I, Stage8_R, Stage8_I);
	fft_stage_8(Stage8_R, Stage8_I, Stage9_R, Stage9_I);
	fft_stage_9(Stage9_R, Stage9_I, Stage10_R, Stage10_I);
	fft_stage_10(Stage10_R, Stage10_I, OUT_R, OUT_I);
}

#endif
