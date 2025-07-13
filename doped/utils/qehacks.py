# hacks.py
import numpy as np
from pymatgen.electronic_structure.core import Spin

class PymatgenEspressoHacks:
    """
    Hacks for pymatgen.io.qe.PWxml to enable setting read-only properties
    like `projected_eigenvalues` and `projected_magnetisation`.
    """

    @staticmethod
    def make_property_settable(cls, prop_name, backing_attr=None):
        """
        Dynamically replace a read-only property on a class with a read-write version
        that allows manual override via a hidden attribute.

        Parameters:
        - cls: The class to patch (e.g., PWxml)
        - prop_name: The name of the property to patch
        - backing_attr: Optional; name of the internal attribute to store override value
        """
        prop = getattr(cls, prop_name, None)
        if not isinstance(prop, property):
            raise TypeError(f"{prop_name} is not a property on {cls}.")

        fget = prop.fget
        backing_attr = backing_attr or f"_{prop_name}_manual"

        def getter(self):
            return getattr(self, backing_attr, fget(self))

        def setter(self, value):
            setattr(self, backing_attr, value)

        setattr(cls, prop_name, property(getter, setter))

    @classmethod
    def patch_pwxml_properties(cls):
        """
        Applies all useful patches to PWxml so these attributes become writable.
        """
        from pymatgen.io.espresso.outputs import PWxml

        properties_to_patch = [
            "projected_eigenvalues",
            "projected_magnetisation",
            #"atomic_states",
            #"kpoints_opt_props",  # Others you may want to override
        ]

        for prop in properties_to_patch:
            try:
                cls.make_property_settable(PWxml, prop)
            except Exception as e:
                print(f"[WARN] Could not patch '{prop}': {e}")

#    @classmethod
#    def atomic_states_to_projdict(cls, atomic_states):
#        """
#        Converts a list of AtomicState objects into a dict[Spin, np.ndarray]
#        with shape (n_states, nbands, nkpts, n_proj), matching projected_eigenvalues.
#        
#        Assumes each atomic_state.projections[spin] has shape (nbands, nkpts, n_proj),
#        where n_proj = 1 typically per state.
#        """
#
#        print("LEN ATOMIC STATES:", len(atomic_states))
#
#        if not atomic_states:
#            return {}
#
#        spin_channels = list(atomic_states[0].projections.keys())
#        nbands, nkpts = atomic_states[0].projections[spin_channels[0]].shape[:2]
#        
#        proj_dict = {spin: [] for spin in spin_channels}
#
#        for spin in spin_channels:
#            for state in atomic_states:
#                proj = state.projections.get(spin)
#                if proj is None:
#                    raise ValueError(f"Missing projection for spin {spin} in state {state.state_i}")
#                # Ensure 3D shape: (nbands, nkpts, 1)
#                if proj.ndim == 2:
#                    proj = proj[..., np.newaxis]
#                proj_dict[spin].append(proj)
#
#            # Stack into shape: (n_states, nbands, nkpts, 1) → transpose if needed
#            proj_dict[spin] = np.stack(proj_dict[spin], axis=0)
#
#        return proj_dict
#
