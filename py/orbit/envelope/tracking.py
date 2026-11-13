from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from orbit.core.bunch import Bunch
from orbit.core.teapot_base import MatrixGenerator

from .envelope import Envelope
from .matrix_fitting import fit_node_transfer_matrix
from .matrix import get_node_matrix
from .sync_particle import SyncParticleParams

if TYPE_CHECKING:
    from orbit.lattice import AccActionsContainer
    from orbit.lattice import AccLattice
    from orbit.lattice import AccNode


ParamsDict = dict[str, Any]


@dataclass
class NodeMap:
    matrix: np.ndarray | None
    sync_params: SyncParticleParams
    time_advance: float

    def update_sync_params(self, sync_params: SyncParticleParams) -> None:
        sync_params.kin_energy = self.sync_params.kin_energy
        sync_params.mass = self.sync_params.mass
        sync_params.charge = self.sync_params.charge
        sync_params.time += self.time_advance


class EnvelopeTracker:
    """Track beam moments using analytic maps or maps fitted with a reusable bunch."""

    def __init__(
        self,
        lattice: AccLattice,
        index_start: int = 0,
        index_stop: int | None = None,
        fit: bool = True,
        fallback: bool = True,
    ) -> None:
        self.lattice = lattice

        self.index_start = index_start
        self.index_start = max(self.index_start, 0)

        self.index_stop = index_stop
        if self.index_stop is None or self.index_stop < 0:
            self.index_stop = len(self.lattice.getNodes())
        else:
            self.index_stop = min(self.index_stop + 1, len(self.lattice.getNodes()))

        self.fit = fit
        self.fallback = fallback
        
        self._node_maps = None
        # Allocate only when fitting (including analytic fallback), and reuse
        # across node parts and successive calls to track/precompute.
        self.bunch: Bunch | None = None
        self._matrix_generator = None

    def track(
        self,
        envelope: Envelope,
        sc: str | None = None,
        params_dict: ParamsDict | None = None,
        action_container: AccActionsContainer | None = None,
        return_matrix: bool = False,
        store_maps: bool = False,
    ) -> np.ndarray | None:

        from orbit.lattice import AccActionsContainer
        from orbit.lattice import AccNode

        if sc not in (None, False, "2d", "3d"):
            raise ValueError(f"Invalid envelope space charge option `{sc}`")

        if params_dict is None:
            params_dict = {}

        if action_container is None:
            action_container = AccActionsContainer()

        params_dict.update(
            envelope=envelope,
            lattice=self.lattice,
            action_container=action_container
        )
        params_dict.setdefault("path_length", 0.0)
        if hasattr(self.lattice, "getUseRealCharge"):
            params_dict.setdefault("useCharge", self.lattice.getUseRealCharge())

        total_matrix = None
        if return_matrix:
            total_matrix = np.eye(7)

        node_maps = {} if store_maps else self._node_maps
        identity_matrix = np.eye(7) if store_maps else None

        def apply_matrix(matrix: np.ndarray | None) -> None:
            nonlocal total_matrix
            if matrix is not None:
                envelope.transform(matrix)
                if total_matrix is not None:
                    total_matrix = matrix @ total_matrix

        def push_envelope(params_dict: ParamsDict) -> None:
            node = params_dict["node"]
            part_index = node.getActivePartIndex()
            if sc:
                length = node.getLength(part_index)
                if length > 0:
                    apply_matrix(envelope.sc_matrix(length, model=sc))
            if node_maps is not None and not store_maps:
                node_map = node_maps[node, part_index]
            else:
                node_map = self._get_node_map(params_dict)
                if store_maps:
                    if node_map.matrix is not None and np.array_equal(node_map.matrix, identity_matrix):
                        node_map.matrix = None
                    node_maps[node, part_index] = node_map
            node_map.update_sync_params(envelope.sync_params)
            apply_matrix(node_map.matrix)

        action_container.addAction(push_envelope, AccNode.BODY)
        try:
            if self.index_start < self.index_stop:
                self.lattice.trackActions(
                    action_container,
                    params_dict,
                    self.index_start,
                    self.index_stop - 1,
                )
        finally:
            action_container.removeAction(push_envelope, AccNode.BODY)
        if store_maps:
            self._node_maps = node_maps
        return total_matrix

    def precompute(self, envelope: Envelope, params_dict: ParamsDict | None = None) -> None:
        self.track(
            envelope=envelope.copy(),
            params_dict=None if params_dict is None else params_dict.copy(),
            store_maps=True,
        )

    def _get_node_map(self, params_dict: ParamsDict) -> NodeMap:
        node = params_dict["node"]
        sync_params = params_dict["envelope"].sync_params
        reference_params = replace(sync_params)
        start_time = sync_params.time
        part_index = node.getActivePartIndex()
        use_fit = self.fit

        if not use_fit:
            try:
                matrix = get_node_matrix(node, reference_params, part_index=part_index)
            except NotImplementedError:
                if not self.fallback:
                    raise
                use_fit = True

        if use_fit:
            if self.bunch is None:
                self.bunch = Bunch()
            sync_params.update_bunch(self.bunch)
            if self._matrix_generator is None:
                self._matrix_generator = MatrixGenerator()
            matrix = fit_node_transfer_matrix(
                node=node,
                bunch=self.bunch,
                part_index=part_index,
                parent_node=params_dict["parentNode"],
                matrix_generator=self._matrix_generator,
                params_dict=params_dict,
            )
            reference_params = SyncParticleParams.from_bunch(self.bunch)

        return NodeMap(
            matrix=matrix,
            sync_params=reference_params,
            time_advance=reference_params.time - start_time,
        )
