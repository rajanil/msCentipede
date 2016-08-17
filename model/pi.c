
#include "footprint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_errno.h>
#include <omp.h>

void pi_function( double* x, double* f, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J, long T )
{

	long idx, n, j, r, l, start, end, L;
	double F, alpha, beta;

	L = powl(2,J)-1;
    omp_set_num_threads(T);

    // loop over scales
    for (j=0; j<J; j++) {

        start = powl(2,j)-1;
        end = powl(2,j+1)-1;
        // loop over locations
        for (l=start; l<end; l++) {

            alpha = x[l] * tau[j];
            beta = (1-x[l]) * tau[j];

            // loop over samples
            #pragma omp parallel for private (r, n, F, idx)
            for (n=0; n<N; n++){

                F = 0.;
                idx = n*L*R + l*R;

                // loop over replicates
                for (r=0; r<R; r++){

                    #pragma omp atomic
                    F += gsl_sf_lngamma(left[idx+r]+alpha) + 
                         gsl_sf_lngamma(total[idx+r]-left[idx+r]+beta);
                }

                #pragma omp atomic
                f[0] += zeta[n]*F;
            }

            #pragma omp atomic
            f[0] -= zetasum * R * (gsl_sf_lngamma(alpha) + gsl_sf_lngamma(beta));
        }
   	}
}

void pi_gradient( double* x, double* Df, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J, long T )
{

    int j;
	long idx, n, r, l, start, end, L;
	double df, alpha, beta;

    L = powl(2,J)-1;
    omp_set_num_threads(T);

    // loop over scales
    for (j=0; j<J; j++) {

        start = powl(2,j)-1;
        end = powl(2,j+1)-1;
        // loop over locations
        for (l=start; l<end; l++) {

            alpha = x[l] * tau[j];
            beta = (1-x[l]) * tau[j];

            // loop over samples
            #pragma omp parallel for private (r, n, df, idx)
            for (n=0; n<N; n++){

                idx = n*L*R + l*R;
                df = 0.;

                // loop over replicates
                for (r=0; r<R; r++){
                    #pragma omp atomic
                    df += gsl_sf_psi(left[idx+r]+alpha) - 
                          gsl_sf_psi(total[idx+r]-left[idx+r]+beta);
                }

                #pragma omp atomic            
                Df[l] += zeta[n] * df;
            }

            Df[l] = tau[j] * (Df[l] - zetasum * R * (gsl_sf_psi(alpha) - gsl_sf_psi(beta)));
        }
    }
}

void pi_hessian( double* x, double* Hf, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J, long T )
{

    int j;
	long idx, n, r, l, start, end, L;
	double hf, alpha, beta;

	L = powl(2,J)-1;
    omp_set_num_threads(T);

    // loop over scales
    for (j=0; j<J; j++) {

        start = powl(2,j)-1;
        end = powl(2,j+1)-1;
        // loop over locations
        for (l=start; l<end; l++) {

            alpha = x[l] * tau[j];
            beta = (1-x[l]) * tau[j];

            // loop over samples
            #pragma omp parallel for private (r, n, hf, idx)
            for (n=0; n<N; n++){

                idx = n*L*R + l*R;
                hf = 0.;

                // loop over replicates
                for (r=0; r<R; r++){
                    #pragma omp atomic
                    hf += gsl_sf_psi_1(left[idx+r]+alpha) + 
                          gsl_sf_psi_1(total[idx+r]-left[idx+r]+beta);
                }

                #pragma omp atomic
                Hf[l*L+l] += zeta[n] * hf;
            }

            Hf[l*L+l] = tau[j] * tau[j] * (Hf[l*L+l] - zetasum * R * (gsl_sf_psi_1(alpha) + gsl_sf_psi_1(beta)));
        }
    }
}

