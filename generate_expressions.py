from posthoceval.model_generation import generate_additive_expression, valid_variable_domains
import sympy as S

symbols = S.symbols('x1:101', real=True)
tries = 0
while True:
    tries += 1
    try:
        expr = generate_additive_expression(symbols, pct_nonlinear=0, n_interaction=0, interaction_ord=(2, 3), nonlinear_multiplier=2, nonlinear_skew=1)
        domains = valid_variable_domains(expr, fail_action='error')
    except (RuntimeError, KeyboardInterrupt):  # allow for keyboard interrupts in case of long generation time (e.g. periodicity infinite recursion)
        pass  # TODO modify my code surrounding continuous_domain & time it out? Or check if periodicity function in stack too long?
        # TODO or inject wrapper of periodicity that kills itself after a bit?
    else:
        break
print(f'Generated valid expression in {tries} tries.')
S.pprint(expr, num_columns=180)
domains
