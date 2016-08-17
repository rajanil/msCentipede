
#include "footprint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_errno.h>
#include <omp.h>

void tau_function( double* x, double* f, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J, long T )
{

    int j;
	long idx, n, r, l, start, end, L;
	double F, alpha, beta;

    L = powl(2,J)-1;
    omp_set_num_threads(T);

    // loop over scales and locations
    for (j=0; j<J; j++) {

        start = powl(2,j)-1;
        end = powl(2,j+1)-1;
        // loop over locations
        for (l=start; l<end; l++) {

            alpha = pi[l] * x[j];
            beta = (1.-pi[l]) * x[j];

            // loop over samples
            #pragma omp parallel for private (n, r, F, idx)
            for (n=0; n<N; n++){

                F = 0.;
                idx = n*L*R + l*R;

                // loop over replicates
                for (r=0; r<R; r++){

                    #pragma omp atomic
                    F += gsl_sf_lngamma(left[idx+r]+alpha) + 
                         gsl_sf_lngamma(total[idx+r]-left[idx+r]+beta) - 
                         gsl_sf_lngamma(total[idx+r]+x[j]);
                }

                #pragma omp atomic
                f[0] += zeta[n]*F;
            }

            #pragma omp atomic
            f[0] += zetasum * R * (gsl_sf_lngamma(x[j]) - gsl_sf_lngamma(alpha) - gsl_sf_lngamma(beta));
        }
   	}
}

void tau_gradient( double* x, double* Df, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J, long T )
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

            alpha = pi[l] * x[j];
            beta = (1.-pi[l]) * x[j];

            // loop over samples
            #pragma omp parallel for private (n, r, df, idx)
            for (n=0; n<N; n++){

                idx = n*L*R + l*R;
                df = 0.;

                // loop over replicates
                for (r=0; r<R; r++){
                    #pragma omp atomic
                    df += pi[l] * gsl_sf_psi(left[idx+r]+alpha) + 
                         (1-pi[l]) * gsl_sf_psi(total[idx+r]-left[idx+r]+beta) - 
                         gsl_sf_psi(total[idx+r]+x[j]);
                }

                #pragma omp atomic
                Df[j] += zeta[n] * df;
            }

            #pragma omp atomic
            Df[j] += zetasum * R * (gsl_sf_psi(x[j]) - pi[l] * gsl_sf_psi(alpha) - 
                        (1-pi[l]) * gsl_sf_psi(beta));
        }
    }
}

void tau_hessian( double* x, double* Hf, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J, long T )
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

            alpha = pi[l] * x[j];
            beta = (1.-pi[l]) * x[j];

            // loop over samples
            #pragma omp parallel for private (n, r, hf, idx)
            for (n=0; n<N; n++){

                idx = n*L*R + l*R;
                hf = 0.;

                // loop over replicates
                for (r=0; r<R; r++){
                    #pragma omp atomic
                    hf += pi[l] * pi[l] * gsl_sf_psi_1(left[idx+r]+alpha) + 
                         (1-pi[l]) * (1-pi[l]) * gsl_sf_psi_1(total[idx+r]-left[idx+r]+beta) - 
                         gsl_sf_psi_1(total[idx+r]+x[j]);
                }
            
                #pragma omp atomic
                Hf[j*J+j] += zeta[n] * hf;
            }

            #pragma omp atomic
            Hf[j*J+j] += zetasum * R * (gsl_sf_psi_1(x[j]) - pi[l] * pi[l] * gsl_sf_psi_1(alpha) - 
                        (1-pi[l]) * (1-pi[l]) * gsl_sf_psi_1(beta));

        }
    }
}

