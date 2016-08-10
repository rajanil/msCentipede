
#include "footprint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_errno.h>

void tau_function( double* x, double* f, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J )
{

    int j;
	long idx, n, r, l, start, end, L;
	double F;
	double *alpha, *beta;

    L = powl(2,J+1)-1;

	// initialize parameters for beta distribution
	alpha = (double*) malloc(L * sizeof(double));
	beta = (double*) malloc(L * sizeof(double));
	// loop over scales
	for (j=0; j<J; j++) {
		start = powl(2,j)-1;
        end = powl(2,j+1)-1;
        // loop over locations
        for (l=start; l<end; l++) {
        	alpha[l] = pi[l] * x[j];
        	beta[l] = (1.-pi[l]) * x[j];
        }
    }

    // loop over scales and locations
    for (l=0; l<L; l++) {

    	frexp(l, &j);

    	// loop over samples
    	for (n=0; n<N; n++){

    		F = 0.;
    		idx = n*L*R + l*R;

    		// loop over replicates
    		for (r=0; r<R; r++){

    			F += gsl_sf_lngamma(left[idx+r]+alpha[l]) + 
    				 gsl_sf_lngamma(total[idx+r]-left[idx+r]+beta[l]) - 
    				 gsl_sf_lngamma(total[idx+r]+x[j]);
    		}

    		f[0] = f[0] + zeta[n]*F;
    	}

    	f[0] = f[0] + zetasum * R * (gsl_sf_lngamma(x[j]) - gsl_sf_lngamma(alpha[l]) - gsl_sf_lngamma(beta[l]));
   	}
   	free( alpha );
   	free( beta );
}

void tau_gradient( double* x, double* Df, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J )
{

    int j;
	long idx, n, r, l, start, end, L;
	double df;
	double *alpha, *beta;

	L = powl(2,J+1)-1;

	// initialize parameters for beta distribution
	alpha = (double*) malloc(L * sizeof(double));
	beta = (double*) malloc(L * sizeof(double));
	// loop over scales
	for (j=0; j<J; j++) {
		start = powl(2,j)-1;
        end = powl(2,j+1)-1;
        // loop over locations
        for (l=start; l<end; l++) {
        	alpha[l] = pi[l] * x[j];
        	beta[l] = (1.-pi[l]) * x[j];
        }
    }

    // loop over scales and locations
    for (l=0; l<L; l++) {

    	frexp(l, &j);

    	// loop over samples
    	for (n=0; n<N; n++){

    		idx = n*L*R + l*R;
    		df = 0.;

    		// loop over replicates
    		for (r=0; r<R; r++){
            	df += pi[l] * gsl_sf_psi(left[idx+r]+alpha[l]) + 
    				 (1-pi[l]) * gsl_sf_psi(total[idx+r]-left[idx+r]+beta[l]) - 
    				 gsl_sf_psi(total[idx+r]+x[j]);
            }
        
        	Df[j] = Df[j] + zeta[n] * df;
        }

        Df[j] = Df[j] + zetasum * R * (gsl_sf_psi(x[j]) - pi[l] * gsl_sf_psi(alpha[l]) - 
        			(1-pi[l]) * gsl_sf_psi(beta[l]));
    }

    free( alpha );
    free( beta );
}

void tau_hessian( double* x, double* Hf, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J )
{

    int j;
	long idx, n, r, l, start, end, L;
	double hf;
	double *alpha, *beta;

	L = powl(2,J+1)-1;

	// initialize parameters for beta distribution
	alpha = (double*) malloc(L * sizeof(double));
	beta = (double*) malloc(L * sizeof(double));
	// loop over scales
	for (j=0; j<J; j++) {
		start = powl(2,j)-1;
        end = powl(2,j+1)-1;
        // loop over locations
        for (l=start; l<end; l++) {
        	alpha[l] = pi[l] * x[j];
        	beta[l] = (1.-pi[l]) * x[j];
        }
    }

    // loop over scales and locations
    for (l=0; l<L; l++) {

    	frexp(l, &j);

    	// loop over samples
    	for (n=0; n<N; n++){

    		idx = n*L*R + l*R;
    		hf = 0.;

    		// loop over replicates
    		for (r=0; r<R; r++){
            	hf += pi[l] * pi[l] * gsl_sf_psi_1(left[idx+r]+alpha[l]) + 
    				 (1-pi[l]) * (1-pi[l]) * gsl_sf_psi_1(total[idx+r]-left[idx+r]+beta[l]) - 
    				 gsl_sf_psi_1(total[idx+r]+x[j]);
            }
        
        	Hf[j*J+j] = Hf[j*J+j] + zeta[n] * hf;
        }

        Hf[j] = Hf[j] + zetasum * R * (gsl_sf_psi_1(x[j]) - pi[l] * pi[l] * gsl_sf_psi_1(alpha[l]) - 
        			(1-pi[l]) * (1-pi[l]) * gsl_sf_psi_1(beta[l]));

    }

    free( alpha );
    free( beta );
}

