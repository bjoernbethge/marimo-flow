# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pina-mathlab",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # PINA: Basic Problem Setup
        
        This snippet demonstrates how to define a PINA problem with domain,
        equations, and boundary conditions for solving differential equations.
        """
    )
    return


@app.cell
def _():
    from marimo_flow.core import ProblemManager
    
    # Create a Poisson problem on the unit square using ProblemManager
    problem_class = ProblemManager.create_poisson_problem()
    problem = problem_class()
    
    return ProblemManager, problem, problem_class


@app.cell
def _(problem):
    # Inspect problem structure
    print(f"Domain: {problem.spatial_domain}")
    print(f"Number of equations: {len(problem.equations)}")
    print(f"Number of conditions: {len(problem.conditions)}")
    print(f"Condition names: {list(problem.conditions.keys())}")
    
    return


if __name__ == "__main__":
    app.run()
