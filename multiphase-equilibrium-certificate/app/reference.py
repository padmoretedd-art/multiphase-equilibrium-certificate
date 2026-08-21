import math
import numpy as np
from scipy.optimize import least_squares
from typing import Dict, Optional
from app.chemistry import ChemicalSystem
from app.validation import EquilibriumValidator


class EquilibriumSolver:
    """Solves non-linear multiphase chemical equilibrium using log-concentration state variables."""

    def __init__(self, ftol: float = 1e-13, xtol: float = 1e-13, max_nfev: int = 20000):
        self.ftol = ftol
        self.xtol = xtol
        self.max_nfev = max_nfev

    def solve(self, system: ChemicalSystem, initial_guess: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        species_list = list(system.species.keys())
        n_sp = len(species_list)
        sp_to_idx = {sp: i for i, sp in enumerate(species_list)}

        elem_list = list(system.initial_totals.keys())
        n_elem = len(elem_list)

        A_elem = np.zeros((n_elem, n_sp))
        b_elem = np.array([system.initial_totals[e] for e in elem_list])
        for i, elem in enumerate(elem_list):
            for sp_id, sp in system.species.items():
                A_elem[i, sp_to_idx[sp_id]] = sp.elements.get(elem, 0.0)

        charge_vec = np.array([system.species[sp].charge for sp in species_list])
        validator = EquilibriumValidator()

        def residuals(ln_c: np.ndarray) -> np.ndarray:
            c = np.exp(ln_c)
            concs = {sp: float(c[sp_to_idx[sp]]) for sp in species_list}
            res = []

            # 1. Chemical Reaction Equilibrium Residuals
            for rxn in system.reactions:
                if rxn.reaction_type == "precipitation":
                    solid_sp = [sp for sp in rxn.stoichiometry if system.species[sp].phase == "solid"]
                    if solid_sp:
                        solid_c = concs.get(solid_sp[0], 0.0)
                        log_res = rxn.calculate_log_residual(concs)
                        if solid_c <= 1e-10 and log_res < 0:
                            res.append(0.0)
                            continue
                res.append(rxn.calculate_log_residual(concs))

            # 2. Elemental Conservation Residuals (weighted heavily to enforce strictly)
            computed_totals = A_elem @ c
            for i, target in enumerate(b_elem):
                scale = max(1e-4, abs(target))
                res.append(10000.0 * (computed_totals[i] - target) / scale)

            # 3. Charge Conservation Residual
            computed_charge = np.dot(charge_vec, c)
            charge_scale = max(1e-4, abs(system.total_charge_initial))
            res.append(10000.0 * (computed_charge - system.total_charge_initial) / charge_scale)

            return np.array(res, dtype=float)

        best_sol = None
        best_cost = float("inf")

        init_concs = [
            np.full(n_sp, -5.0),
            np.full(n_sp, -3.0),
            np.full(n_sp, -8.0),
            np.full(n_sp, -1.0),
        ]

        if initial_guess:
            custom_init = np.array([math.log(max(1e-12, initial_guess.get(sp, 1e-3))) for sp in species_list])
            init_concs.insert(0, custom_init)

        for x0 in init_concs:
            try:
                res = least_squares(
                    residuals,
                    x0,
                    bounds=(-35.0, 5.0),
                    method="trf",
                    ftol=self.ftol,
                    xtol=self.xtol,
                    max_nfev=self.max_nfev,
                )
                c_opt = np.exp(res.x)
                sol = {sp: float(c_opt[i]) for i, sp in enumerate(species_list)}
                val_res = validator.validate(system, {"concentrations": sol})

                if val_res.is_valid:
                    return sol

                if res.cost < best_cost:
                    best_cost = res.cost
                    best_sol = sol
            except Exception:
                continue

        if best_sol is None:
            raise RuntimeError("Equilibrium solver failed to converge to a valid solution.")

        return best_sol
