
# numerical libraries
import numpy as np

# optimization libraries
import cvxopt as cvx
from cvxopt import solvers

# custom libraries
import utils

# suppress optimizer output
solvers.options['show_progress'] = False

def optimize(xo, function_gradient, function_gradient_hessian, dict args):
    """Calls the appropriate nonlinear convex optimization solver 
    in the package `cvxopt` to find optimal values for the relevant
    parameters, given subroutines that evaluate a function, 
    its gradient, and hessian, this subroutine 

    Arguments
    	xo : numpy.ndarray
    		 initial value for optimization

        function_gradient : function object
        					evaluates the function and the gradient
        					at the specified parameter values

        function_gradient_hessian : function object
        							evaluates the function, gradient and hessian
        							at the specified parameter values

        args : dict
        	   dictionary of additional arguments and
        	   optimization constraints

    """

    def F(x=None, z=None):
        """A subroutine that the cvxopt package can call to get 
        values of the function, gradient and hessian during
        optimization.
        """

        if x is None:
            return 0, cvx.matrix(x_init)

        xx = np.array(x).ravel()

        if z is None:

            # compute likelihood function and gradient
            f, Df = function_gradient(xx, args)

            # check for infs and nans in function and gradient
            if np.isnan(f) or np.isinf(f):
                f = np.array([utils.MAX], dtype=float)
            else:
                f = -1*f.astype('float')
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * utils.MAX * np.ones((1,xx.size), dtype=float)
            else:
                Df = -1 * Df

            return cvx.matrix(f), cvx.matrix(Df)

        else:

            # compute likelihood function, gradient, and hessian
            f, Df, Hf = function_gradient_hessian(xx, args)

            # check for infs and nans in function and gradient
            if np.isnan(f) or np.isinf(f):
                f = np.array([utils.MAX], dtype=float)
            else:
                f = -1*f.astype('float')
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * utils.MAX * np.ones((1,xx.size), dtype=float)
            else:
                Df = -1 * Df

            Hf = z[0] * Hf
            return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(Hf)

    # warm start for the optimization
    V = xo.size
    x_init = xo.reshape(V,1)

    try:

        # call the optimization subroutine in cvxopt
        if args.has_key('G'):
            # call a constrained nonlinear solver
            solution = solvers.cp(F, G=cvx.matrix(args['G']), h=cvx.matrix(args['h']))
        else:
            # call an unconstrained nonlinear solver
            solution = solvers.cp(F)

        # check if optimal value has been reached
        if solution['status']=='optimal':
            x_final = np.array(solution['x']).ravel()
        else:
            # if optimizer didn't converge,
            # skip this current optimization step. 
            x_final = x_o

    except ValueError:

        # if any parameter becomes Inf or Nan during optimization,
        # skip this current optimization step.
        x_final = x_o

    return x_final