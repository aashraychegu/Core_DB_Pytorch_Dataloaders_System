from typing import Any
from .._filepaths._filepaths import CoRe_DB_path
import h5py as h5
from watpy.coredb.coredb import *
import numpy as np
import pathlib as p
import math
from ..utilites._preprocessing import *


class h5Finder:
    def __init__(
        self,
        path=CoRe_DB_path,
        selection_attributes=["id_eos", "id_mass_starA", "id_mass_starB"],
        device="cpu",
        sync=False,
        all_radii=False,
        shiftpercents=list(np.array(range(-100, 101, 5)) / 100),
        angles=None,
        distances=[0],
    ) -> None:
        self.path = path
        self.selection_attributes = selection_attributes
        self.device = device
        with HiddenPrints():
            if not self.path.exists():
                self.path.mkdir(exist_ok=False)
            cdb = CoRe_db(self.path)
            if sync:
                cdb.sync(verbose=False, lfs=True)
            self.sims = cdb.sim
        self.pspace = []
        self.eoss = []
        self.device = device
        with HiddenPrints():
            if not self.path.exists():
                self.path.mkdir(exist_ok=False)
            cdb = CoRe_db(self.path)
            if sync:
                cdb.sync(verbose=False, lfs=True)
            self.sims = cdb.sim

        for sim_key in self.sims:
            sim = self.sims[sim_key]
            for run_key in sim.run:
                run = sim.run[run_key]
                current_h5_filepath = p.Path(self.sims[sim_key].run[run_key].data.path)
                current_h5_file = h5.File(current_h5_filepath / "data.h5", "r")
                current_rh_waveforms = [
                    i for i in current_h5_file.keys() if i == "rh_22"
                ]
                for selected_wf in current_rh_waveforms:
                    extraction_radii = list(
                        current_h5_file[selected_wf].keys()
                    )  # type: ignore
                    if all_radii:
                        for extraction_radius in extraction_radii:
                            self.eoss.append(run.md.data["id_eos"])
                            inserter = (
                                sim_key,
                                run_key,
                                selected_wf,
                                extraction_radius,
                            )
                    else:
                        self.eoss.append(run.md.data["id_eos"])
                    inserter_base = [
                        sim_key,
                        run_key,
                        selected_wf,
                        extraction_radii[-1],
                    ]

                    for i in shiftpercents:
                        for j in distances:
                            if angles is not None:
                                for k in angles:
                                    self.pspace.append(inserter_base + [i, j, k])
                            else:
                                self.pspace.append(inserter_base + [i, j])
        # ----
        self.datapoints = self.pspace
        self.unique_eos, self.counts_per_eos = np.unique(
            np.array(self.eoss), return_counts=True
        )
        self.unique_eos = list(self.unique_eos)
        self.eos_to_index_map = {i: self.unique_eos.index(i) for i in self.unique_eos}
        self.index_to_eos_map = {self.unique_eos.index(i): i for i in self.unique_eos}
        self.number_of_eos = len(self.eoss)

    def get_datapoints(self, *args: Any, **kwds: Any) -> Any:
        return (
            np.array(self.datapoints),
            self.eos_to_index_map,
            [
                self.unique_eos,
                self.eos_to_index_map,
                self.index_to_eos_map,
                self.number_of_eos,
            ],
        )
