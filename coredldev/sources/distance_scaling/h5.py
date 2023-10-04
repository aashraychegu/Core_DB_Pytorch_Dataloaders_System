from ..._filepaths._filepaths import CoRe_DB_path
import h5py as h5
from watpy.coredb.coredb import *
import numpy as np
import pathlib as p
import math
from ...utilites._preprocessing import *
from ..._resources.eos_to_index_map import eos_to_index_map


class h5Source:
    def __init__(
        self,
        path=CoRe_DB_path,
        selection_attributes=["id_eos", "id_mass_starA", "id_mass_starB",],
        device="cpu",
        sync=False,
        eos_to_index_map=eos_to_index_map[0],
    ) -> None:
        self.eos_to_index_map = eos_to_index_map
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

    def __getitem__(self, index: int):
        return self.preprocess(*self.retrieve(index))

    def retrieve(self, psl:dict):
        data = self.sims[psl["sim_key"]].run[psl["run_key"]]
        h5path = p.Path(data.data.path) / "data.h5"
        metadata = {i: data.md.data[i] for i in self.selection_attributes}
        data = h5.File(h5path, "r")[psl["selected_wf"]][psl["extraction_radii"]]  # type: ignore
        pm_time = data[:, -1] *4.975e-6 # type: ignore
        hplus, hcross, pm_time = cut_at_lowest_envelope(data[:, 1], data[:, 2], pm_time=pm_time)  # type: ignore
        sam_p = (pm_time[-1] - pm_time[0]) / len(pm_time)  # type: ignore
        return hplus, hcross, pm_time, metadata, sam_p, psl["shifts"], psl["distance"], psl["angles"], psl["extraction_radii"]

    def preprocess(self, hplus, hcross, pm_time, params, sam_p, percent_shift, rescale_to_radii, angle,current_extraction_radii):
        lts = len(hplus)
        if lts < pad_to_for_planck_window:
            hplus = np.concatenate(
                (
                    np.zeros(math.floor(pad_to_for_planck_window - lts)),
                    hplus,
                    np.zeros(math.ceil(pad_to_for_planck_window - lts)),
                ),
                axis=0,
            )
            hcross = np.concatenate(
                (
                    np.zeros(math.floor(pad_to_for_planck_window - lts)),
                    hcross,
                    np.zeros(math.ceil(pad_to_for_planck_window - lts)),
                ),
                axis=0,
            )
            print("contingency")
        clean_xrad = float(current_extraction_radii.split("_")[-1].split(".")[0][1:])
        return dict(
            hplus = hplus,
            hcross = hcross,
            pm_time = pm_time,
            params = dict(
                eos = self.eos_to_index_map[params["id_eos"]],
                mass_starA = float(params["id_mass_starA"]),
                mass_starB = float(params["id_mass_starB"]),
                percent_shift = float(percent_shift),
                rescale_to_radii = float(rescale_to_radii),
                angle = angle,
                sam_p = float(sam_p),
                current_extraction_radius = clean_xrad,
            ),
        )
