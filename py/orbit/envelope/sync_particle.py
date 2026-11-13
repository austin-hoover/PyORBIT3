from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from orbit.utils.consts import mass_proton

if TYPE_CHECKING:
    from orbit.core.bunch import Bunch


@dataclass
class SyncParticleParams:
    kin_energy: float = 0.0
    mass: float = mass_proton
    charge: float = 1.0
    time: float = 0.0

    @property
    def gamma(self) -> float:
        return 1.0 + self.kin_energy / self.mass

    @property
    def momentum(self) -> float:
        return math.sqrt(self.kin_energy * (self.kin_energy + 2.0 * self.mass))

    @property
    def beta(self) -> float:
        return self.momentum / (self.kin_energy + self.mass)

    @classmethod
    def from_bunch(cls, bunch: Bunch) -> SyncParticleParams:
        """Snapshot the reference state without retaining the bunch."""
        sync_part = bunch.getSyncParticle()
        return cls(
            kin_energy=sync_part.kinEnergy(),
            mass=bunch.mass(),
            charge=bunch.charge(),
            time=sync_part.time(),
        )

    def update_bunch(self, bunch: Bunch) -> None:
        """Set the bunch reference state before fitting or particle tracking."""
        bunch.mass(self.mass)
        bunch.charge(self.charge)
        sync_part = bunch.getSyncParticle()
        sync_part.kinEnergy(self.kin_energy)
        sync_part.time(self.time)
