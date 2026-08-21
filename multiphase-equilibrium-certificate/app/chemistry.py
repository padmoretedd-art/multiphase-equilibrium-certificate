import math
from typing import Dict, Any, Optional


class Species:
    def __init__(self, species_id: str, formula: str, charge: int, phase: str, elements: Dict[str, float]):
        self.species_id = species_id
        self.formula = formula
        self.charge = charge
        self.phase = phase
        self.elements = elements

    def activity_coefficient(self, ionic_strength: float = 0.0, A: float = 0.509) -> float:
        """Calculates ionic activity coefficient using Davies equation."""
        if self.phase != "aqueous" or self.charge == 0 or ionic_strength <= 0:
            return 1.0
        sqrt_I = math.sqrt(ionic_strength)
        log_gamma = -A * (self.charge ** 2) * ((sqrt_I / (1.0 + sqrt_I)) - 0.3 * ionic_strength)
        return 10 ** log_gamma

    def to_dict(self) -> Dict[str, Any]:
        return {
            "species_id": self.species_id,
            "formula": self.formula,
            "charge": self.charge,
            "phase": self.phase,
            "elements": self.elements,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Species':
        """Deserializes a dictionary into a Species instance."""
        return cls(
            species_id=data["species_id"],
            formula=data["formula"],
            charge=data["charge"],
            phase=data["phase"],
            elements=data["elements"],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Species):
            return False
        return (
            self.species_id == other.species_id
            and self.formula == other.formula
            and self.charge == other.charge
            and self.phase == other.phase
            and self.elements == other.elements
        )


class Reaction:
    def __init__(self, reaction_id: str, stoichiometry: Dict[str, float], equilibrium_constant: float, reaction_type: str):
        self.reaction_id = reaction_id
        self.stoichiometry = stoichiometry
        self.equilibrium_constant = equilibrium_constant
        self.reaction_type = reaction_type

    def calculate_quotient(self, concentrations: Dict[str, float]) -> float:
        """Alias for calculate_reaction_quotient."""
        return self.calculate_reaction_quotient(concentrations)

    def calculate_reaction_quotient(self, concentrations: Dict[str, float]) -> float:
        """Ideal concentration reaction quotient Q_c."""
        Q = 1.0
        for sp_id, coeff in self.stoichiometry.items():
            c = max(1e-18, concentrations.get(sp_id, 1e-18))
            Q *= (c ** coeff)
        return Q

    def calculate_activity_quotient(self, concentrations: Dict[str, float], species_dict: Optional[Dict[str, Species]] = None, ionic_strength: float = 0.0, use_activities: bool = False) -> float:
        """Calculates Q_a if use_activities=True, else returns ideal Q_c."""
        if not use_activities or not species_dict or ionic_strength <= 0:
            return self.calculate_reaction_quotient(concentrations)
        
        Q = 1.0
        for sp_id, coeff in self.stoichiometry.items():
            sp = species_dict.get(sp_id)
            c = max(1e-18, concentrations.get(sp_id, 1e-18))
            gamma = sp.activity_coefficient(ionic_strength) if sp else 1.0
            activity = c * gamma if sp and sp.phase == "aqueous" else c
            Q *= (activity ** coeff)
        return Q

    def calculate_log_residual(self, concentrations: Dict[str, float], species_dict: Optional[Dict[str, Species]] = None, ionic_strength: float = 0.0, use_activities: bool = False) -> float:
        """Calculates log10(Q/K) residual."""
        Q_val = self.calculate_activity_quotient(concentrations, species_dict, ionic_strength, use_activities)
        return math.log10(max(1e-18, Q_val)) - math.log10(self.equilibrium_constant)


class ChemicalSystem:
    def __init__(self, species: Dict[str, Species], reactions: list, initial_totals: Dict[str, float], total_charge_initial: float = 0.0):
        self.species = species
        self.reactions = reactions
        self.initial_totals = initial_totals
        self.total_charge_initial = total_charge_initial

    def calculate_elemental_totals(self, concentrations: Dict[str, float]) -> Dict[str, float]:
        """Calculates total moles/concentrations for each element present in the system."""
        totals = {elem: 0.0 for elem in self.initial_totals.keys()}
        for sp_id, sp in self.species.items():
            conc = concentrations.get(sp_id, 0.0)
            for elem, count in sp.elements.items():
                if elem in totals:
                    totals[elem] += count * conc
        return totals

    def calculate_total_charge(self, concentrations: Dict[str, float]) -> float:
        """Calculates total charge in solution sum(z_i * c_i)."""
        return sum(sp.charge * concentrations.get(sp_id, 0.0) for sp_id, sp in self.species.items())

    def calculate_ionic_strength(self, concentrations: Dict[str, float]) -> float:
        """Calculates total ionic strength: I = 0.5 * sum(z_i^2 * c_i)"""
        I = 0.0
        for sp_id, sp in self.species.items():
            if sp.phase == "aqueous":
                c = max(0.0, concentrations.get(sp_id, 0.0))
                I += 0.5 * (sp.charge ** 2) * c
        return I
