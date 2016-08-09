void pi_function( double* x, double* f, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J );

void pi_gradient( double* x, double* Df, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J );

void pi_hessian( double* x, double* Hf, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J );

void tau_function( double* x, double* f, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J );
    
void tau_gradient( double* x, double* Df, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J );
    
void tau_hessian( double* x, double* Hf, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J );