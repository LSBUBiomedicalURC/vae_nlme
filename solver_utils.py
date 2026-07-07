"""
solver_utils.py
----------------

Lets callers swap the QP/MIP solver used by the compiled ParaUpdate/pop_parameter
module away from Gurobi, WITHOUT touching the compiled binary.

Why this works
--------------
pop_parameter.update_pop's covariate-selection step is a Cython-compiled call
of the form `prob.solve(solver=cp.GUROBI)` (confirmed by inspecting the
generated pop_parameter.c). `cp.GUROBI` is just a live attribute lookup on the
`cvxpy` module object, resolved at call time -- it is NOT a constant baked
into the compiled code. So re-pointing `cvxpy.GUROBI` to a different solver
name from Python, before update_pop() is called, changes which solver the
compiled module actually uses. No recompilation needed.

Why this is NOT generally a drop-in fix
----------------------------------------
The covariate-selection problem is a mixed-integer QUADRATIC program: each
parameter's regression introduces a continuous coefficient vector `beta` AND
a boolean indicator vector `z` (big-M-style sparsity constraints) -- i.e. a
genuine MIQP, not a plain QP or a continuous L1/SCAD relaxation. Verified
empirically against this repo's installed cvxpy (1.9.2):
    - ECOS, SCS, CLARABEL, OSQP: cannot solve it at all (no integer-variable
      support).
    - HIGHS (via cvxpy 1.9's HiGHS interface): also rejects it
      ("the solver HIGHS cannot solve this problem") -- cvxpy's current HiGHS
      reduction does not accept this mixed boolean+quadratic problem class,
      even though HiGHS itself has some native MIQP support.
    - GUROBI: works, but the user's license caps total variables (~180),
      which blocks runs with many covariates (z_dim * n_cov can exceed that
      quickly).

The only other solver that handles boolean+quadratic problems through cvxpy
is SCIP (open source, no license restriction, via `pip install pyscipopt`).
If you install it, `set_pop_parameter_solver('SCIP')` will route to it with
no other code changes. We have NOT verified SCIP's performance/scaling on
this problem class in this environment (it is not installed here) -- treat
it as the recommended next thing to try if Gurobi's variable cap is the
actual bottleneck, not as a verified fix.

Usage
-----
    from solver_utils import set_pop_parameter_solver
    set_pop_parameter_solver('GUROBI')   # default, what the binary asks for
    set_pop_parameter_solver('SCIP')     # if pyscipopt is installed
"""

import cvxpy as cp

# Solvers verified (in this repo's env) to be unable to solve pop_parameter's
# MIQP -- listed so callers get an informative error instead of a deep cvxpy
# traceback when they pick one of these by mistake.
_KNOWN_INCOMPATIBLE = {
    'ECOS': "no integer-variable support",
    'SCS': "no integer-variable support",
    'CLARABEL': "no integer-variable support",
    'OSQP': "no integer-variable support",
    'HIGHS': "cvxpy's current HiGHS reduction rejects this mixed boolean+quadratic problem",
    'SCIPY': "LP-only via cvxpy's SciPy interface, no quadratic/integer support",
}


def set_pop_parameter_solver(name, allow_incompatible=False):
    """
    Re-point cvxpy.GUROBI -> cvxpy.<name>, so that pop_parameter.update_pop's
    hardcoded `prob.solve(solver=cp.GUROBI)` call actually uses solver `name`.

    Parameters
    ----------
    name : str
        A cvxpy solver name (e.g. 'GUROBI', 'SCIP'). Must be one of
        `cvxpy.installed_solvers()` (or generally a valid cvxpy solver
        constant) and must support mixed-integer quadratic programs to have
        any chance of solving this repo's covariate-selection step.
    allow_incompatible : bool
        If False (default), raises immediately for solvers in
        `_KNOWN_INCOMPATIBLE` (verified to fail on this problem class) rather
        than letting the user burn a training run before hitting a
        cvxpy.error.SolverError deep inside pop_parameter.update_pop.
    """
    name = name.upper()
    if name == 'GUROBI':
        cp.GUROBI = cp.GUROBI  # no-op, restores default
        return
    if not allow_incompatible and name in _KNOWN_INCOMPATIBLE:
        raise ValueError(
            f"Solver '{name}' is known to be unable to solve pop_parameter's "
            f"mixed-integer QP step in this repo ({_KNOWN_INCOMPATIBLE[name]}). "
            "If you still want to try it (e.g. cvxpy/solver versions changed), "
            "call with allow_incompatible=True / pass --allow_incompatible_solver."
        )
    if not hasattr(cp, name):
        raise ValueError(f"cvxpy has no solver constant '{name}'. "
                          f"Installed solvers: {cp.installed_solvers()}")
    cp.GUROBI = getattr(cp, name)
