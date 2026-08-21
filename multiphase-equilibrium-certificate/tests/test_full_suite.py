import json
import pytest
from app.chemistry import Species, Reaction, ChemicalSystem
from app.reference import EquilibriumSolver
from app.validation import EquilibriumValidator
from app.evaluator import BenchmarkEvaluator


@pytest.fixture
def carbonate_precipitation_system():
    sp_ca = Species("Ca2+", "Ca2+", 2, "aqueous", {"Ca": 1.0})
    sp_co3 = Species("CO32-", "CO32-", -2, "aqueous", {"C": 1.0, "O": 3.0})
    sp_hco3 = Species("HCO3-", "HCO3-", -1, "aqueous", {"H": 1.0, "C": 1.0, "O": 3.0})
    sp_h = Species("H+", "H+", 1, "aqueous", {"H": 1.0})
    sp_caco3_s = Species("CaCO3(s)", "CaCO3", 0, "solid", {"Ca": 1.0, "C": 1.0, "O": 3.0})

    rxn_p = Reaction("ksp", {"CaCO3(s)": -1.0, "Ca2+": 1.0, "CO32-": 1.0}, 3.31131e-9, "precipitation")
    rxn_a = Reaction("ka2", {"HCO3-": -1.0, "H+": 1.0, "CO32-": 1.0}, 4.67735e-11, "acid_base")

    # Consistent physical system: 0.01M Ca2+, 0.01M CO32-, 0.005M H+
    return ChemicalSystem(
        species={
            "Ca2+": sp_ca,
            "CO32-": sp_co3,
            "HCO3-": sp_hco3,
            "H+": sp_h,
            "CaCO3(s)": sp_caco3_s,
        },
        reactions=[rxn_p, rxn_a],
        initial_totals={"Ca": 0.01, "C": 0.01, "O": 0.03, "H": 0.005},
        total_charge_initial=0.005,
    )


def test_precipitation_system_reference(carbonate_precipitation_system):
    solver = EquilibriumSolver()
    sol = solver.solve(carbonate_precipitation_system)

    validator = EquilibriumValidator()
    val_res = validator.validate(carbonate_precipitation_system, {"concentrations": sol})
    assert val_res.is_valid, f"Validation failed with errors: {val_res.errors}"


def test_evaluator_integration(carbonate_precipitation_system):
    evaluator = BenchmarkEvaluator()
    ref_solver = EquilibriumSolver()

    metrics = evaluator.evaluate_solver(
        candidate_id="reference_benchmark",
        solver_fn=ref_solver.solve,
        system=carbonate_precipitation_system,
    )

    assert metrics.is_valid
    assert metrics.max_relative_error_vs_reference < 1e-3
    assert metrics.execution_time_ms > 0.0

    res_dict = metrics.to_dict()
    assert "candidate_id" in res_dict
    assert "validation" in res_dict


def test_validator_rejects_malformed_submission(carbonate_precipitation_system):
    validator = EquilibriumValidator()

    res1 = validator.validate(carbonate_precipitation_system, "not a dict")
    assert not res1.is_valid

    res2 = validator.validate(carbonate_precipitation_system, {"concentrations": {"Ca2+": 0.01}})
    assert not res2.is_valid
    assert any("Missing species" in err for err in res2.errors)

    res3 = validator.validate(
        carbonate_precipitation_system,
        {
            "concentrations": {
                "Ca2+": "invalid",
                "CO32-": 0.01,
                "HCO3-": 0.01,
                "H+": 0.01,
                "CaCO3(s)": 0.01,
            }
        },
    )
    assert not res3.is_valid
    assert any("Invalid non-numeric" in err for err in res3.errors)
