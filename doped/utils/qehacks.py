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

