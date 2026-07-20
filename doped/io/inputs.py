r"""
Calculator-agnostic base class for generating defect calculation inputs.

``DefectsSetBase`` implements the calculator-independent orchestration of
defect calculation input generation -- formatting & naming the input
|DefectEntry|\s, building a per-defect input-set dictionary, and writing the
``<output_path>/<defect name>/<subfolder>`` folder structure (with
serialisation of the defect entries for calculation provenance) -- while each
calculator backend (``doped.io.<calculator>.inputs``) subclasses it to
generate the actual calculation input files (e.g. |DefectsSet| for VASP).

Per-calculation input sets should be built on the ``pymatgen.io.core``
``InputSet``/``InputGenerator`` framework where available for the given
calculator (e.g. ``VaspInputSet`` for VASP, on which
``doped.io.vasp.inputs.DopedDictSet`` is based), so that input files are
generated & written with the standard ``get_input_set()``/``write_input()``
patterns. See the "Adding Support for a New Calculator" docs page for
details.
"""

import contextlib
import copy
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from monty.json import MSONable
from monty.serialization import dumpfn
from pymatgen.util.typing import PathLike
from tqdm import tqdm

from doped.core import DefectEntry
from doped.utils import _doped_obj_properties_methods, get_mp_context, pool_manager

if TYPE_CHECKING:
    from doped.generation import DefectsGenerator


class DefectsSetBase(MSONable, ABC):
    r"""
    Base class for generating calculation input files for a set of
    ``doped``/``pymatgen`` |DefectEntry| objects, for a given calculator.

    Subclasses (``doped.io.<calculator>.inputs``) must implement:

    - :meth:`_defect_input_set`: build the input-set object for a single
      defect entry (e.g. ``DefectRelaxSet`` with VASP), covering the
      calculation stages/subfolders for that calculator's defect workflow.
    - :meth:`_write_defect`: write the calculation input files for a single
      defect (one item of the ``args_list`` built in :meth:`write_files`).

    and may override :meth:`_setup` for any post-input-formatting setup
    (before the per-defect input sets are built).
    """

    _input_set_name: str = "defect input set"  # name of the per-defect input-set class, for messages

    def __init__(
        self,
        defect_entries: "DefectsGenerator | dict[str, DefectEntry] | list[DefectEntry] | DefectEntry",
        **kwargs,
    ):
        r"""
        Format the input ``defect_entries`` and build the per-defect input sets
        (``self.defect_sets``; ``{defect name: input set}``).

        Args:
            defect_entries (|DefectsGenerator|, dict/list of |DefectEntry|\s, or |DefectEntry|):
                Either a |DefectsGenerator| object, or a dictionary/list of
                |DefectEntry|\s, or a single |DefectEntry| object, for which
                to generate calculation input files. If a |DefectsGenerator|
                object or a dictionary (-> ``{defect name: DefectEntry}``),
                the defect folder names will be set equal to ``defect name``.
                If a list or single |DefectEntry| object is provided, the
                defect folder names will be set equal to
                ``DefectEntry.name`` if the ``name`` attribute is set,
                otherwise generated according to the ``doped`` convention
                (see ``doped.generation``).
            **kwargs:
                Additional keyword arguments for the calculator-specific
                input-set generation (stored as ``self.kwargs``).
        """
        self.kwargs = kwargs
        self.defect_entries, self.json_name, self.json_obj = self._format_defect_entries_input(
            defect_entries
        )
        self._setup()

        self.defect_sets = {
            defect_species: self._defect_input_set(defect_entry)
            for defect_species, defect_entry in self.defect_entries.items()
        }
        if not self.defect_sets:
            raise ValueError(
                f"No `{self._input_set_name}` objects created, indicating problems with the "
                f"`{type(self).__name__}` input/creation!"
            )

    def _setup(self) -> None:
        """
        Hook for calculator-specific setup after the input ``defect_entries``
        have been formatted (``self.defect_entries``), but before the per-
        defect input sets are built.
        """

    @abstractmethod
    def _defect_input_set(self, defect_entry: DefectEntry):
        """
        Build the (calculator-specific) input-set object for a single defect
        entry, covering the calculation stages/subfolders of the defect
        workflow for this calculator.
        """

    @staticmethod
    @abstractmethod
    def _write_defect(args: tuple) -> None:
        """
        Write the calculation input files for a single defect, from one item of
        the ``args_list`` built in :meth:`write_files`: ``(defect_species,
        defect_input_set, output_path, bulk, write_kwargs)``.

        A ``staticmethod`` taking a single argument, so that it can be used
        with ``multiprocessing``.
        """

    def _format_defect_entries_input(
        self,
        defect_entries: "DefectsGenerator | dict[str, DefectEntry] | list[DefectEntry] | DefectEntry",
    ) -> "tuple[dict[str, DefectEntry], str, dict[str, DefectEntry] | DefectsGenerator]":
        r"""
        Helper function to format input ``defect_entries`` into a named
        dictionary of |DefectEntry| objects.

        Also returns the name of the JSON file and object to serialise when
        writing the calculation inputs to files. This is the
        |DefectsGenerator| object if ``defect_entries`` is a
        |DefectsGenerator| object, otherwise the dictionary of |DefectEntry|
        objects.

        Args:
            defect_entries (|DefectsGenerator|, dict/list of |DefectEntry|\s, or |DefectEntry|):
                Either a |DefectsGenerator| object, or a dictionary/list of
                |DefectEntry|\s, or a single |DefectEntry| object, for which
                to generate calculation input files.
                If a |DefectsGenerator| object or a dictionary (->
                ``{defect name: DefectEntry}``), the defect folder names will
                be set equal to ``defect name``. If a list or single
                |DefectEntry| object is provided, the defect folder names
                will be set equal to ``DefectEntry.name`` if the ``name``
                attribute is set, otherwise generated according to the
                ``doped`` convention (see ``doped.generation``).
        """
        from doped.generation import name_defect_entries
        from doped.utils.symmetry import _frac_coords_sort_func

        json_filename = "defect_entries.json.gz"  # global statement in case, but should be skipped
        json_obj = defect_entries
        if type(defect_entries).__name__ == "DefectsGenerator":
            defect_entries = cast("DefectsGenerator", defect_entries)
            formula = defect_entries.primitive_structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
            json_filename = f"{formula}_defects_generator.json.gz"
            json_obj = defect_entries
            defect_entries = defect_entries.defect_entries

        elif isinstance(defect_entries, DefectEntry):
            defect_entries = [defect_entries]
        if isinstance(
            defect_entries, list
        ):  # also catches case where defect_entries is a single DefectEntry, from converting to list above
            # need to convert to dict with doped names as keys:
            defect_entry_list = copy.deepcopy(defect_entries)
            with contextlib.suppress(AttributeError, TypeError):  # sort by conventional cell
                # fractional coordinates if these are defined, to aid deterministic naming
                defect_entry_list.sort(key=lambda x: _frac_coords_sort_func(x.conv_cell_frac_coords))

            # figure out which DefectEntry objects need to be named (don't name if already named)
            defect_entries_to_name = [
                defect_entry for defect_entry in defect_entry_list if not hasattr(defect_entry, "name")
            ]
            new_named_defect_entries_dict = name_defect_entries(defect_entries_to_name)
            # set name attribute: (these are names without charges!)
            for defect_name_wout_charge, defect_entry in new_named_defect_entries_dict.items():
                defect_entry.name = (
                    f"{defect_name_wout_charge}_{'+' if defect_entry.charge_state > 0 else ''}"
                    f"{defect_entry.charge_state}"
                )

            # if any duplicate names, crash (and burn, b...)
            if len({defect_entry.name for defect_entry in defect_entry_list}) != len(defect_entry_list):
                raise ValueError(
                    "Some defect entries have the same name, due to mixing of named and unnamed input "
                    "`DefectEntry`s! This would cause defect folders to be overwritten. Please check "
                    "your DefectEntry names and/or generate your defects using DefectsGenerator instead."
                )

            defect_entries = {defect_entry.name: defect_entry for defect_entry in defect_entry_list}
            formula = defect_entry_list[0].defect.structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
            json_filename = f"{formula}_defect_entries.json.gz"
            json_obj = defect_entries

        # check correct format:
        if isinstance(defect_entries, dict) and not all(
            isinstance(defect_entry, DefectEntry) for defect_entry in defect_entries.values()
        ):
            raise TypeError(
                f"Input defect_entries dict must be of the form {{defect_name: DefectEntry}}, got dict "
                f"with values of type {[type(value) for value in defect_entries.values()]} instead"
            )

        if not isinstance(defect_entries, dict):
            raise TypeError(
                f"Input defect_entries must be of type DefectsGenerator, dict, list or DefectEntry, got "
                f"type {type(defect_entries)} instead."
            )

        # ``defect_entries`` validated as a ``{name: DefectEntry}`` dict above:
        return cast(
            "tuple[dict[str, DefectEntry], str, dict[str, DefectEntry] | DefectsGenerator]",
            (defect_entries, json_filename, json_obj),
        )

    def write_files(
        self,
        output_path: PathLike = ".",
        bulk: bool | str = True,
        processes: int | None = None,
        **kwargs,
    ):
        r"""
        Write calculation input files to folders for all defects in
        ``self.defect_entries``, in the ``<output_path>/<defect
        name>/<subfolder>`` folder structure (with subfolders corresponding to
        the calculation stages of the calculator defect workflow).

        The defect entries (``self.json_obj``) are also serialised to
        ``self.json_name`` in ``output_path``, to aid calculation provenance.

        Args:
            output_path (PathLike):
                Folder in which to create the defect calculation folders.
                Default is the current directory (".").
            bulk (bool, str):
                Whether to also write the input files for the reference bulk
                supercell calculation (written once, alongside the defect
                folders; interpretation of non-boolean values is
                calculator-specific). Default is ``True``.
            processes (int):
                Number of processes to use for ``multiprocessing`` for file
                writing. If ``None`` (default), then is dynamically set to the
                optimal value for the number of folders to write.
            **kwargs:
                Additional (calculator-specific) keyword arguments for the
                per-defect file writing (passed through to
                :meth:`_write_defect`).
        """
        args_list = [
            (
                defect_species,
                defect_input_set,
                output_path,
                bulk if i == len(self.defect_sets) - 1 else False,  # write bulk folder(s) for last defect
                kwargs,
            )
            for i, (defect_species, defect_input_set) in enumerate(self.defect_sets.items())
        ]
        if processes is None:  # best setting for number of processes, from testing
            mp = get_mp_context()
            processes = min(round(len(args_list) / 30), mp.cpu_count() - 1)

        if processes > 1:
            with pool_manager(processes) as pool:
                for _ in tqdm(
                    pool.imap(self._write_defect, args_list),
                    total=len(args_list),
                    desc="Generating and writing input files",
                ):
                    pass
        else:
            for args in tqdm(args_list, desc="Generating and writing input files"):
                self._write_defect(args)

        dumpfn(self.json_obj, os.path.join(output_path, self.json_name))

    def __repr__(self):
        """
        Returns a string representation of this defects input set object.
        """
        formula = next(
            iter(self.defect_entries.values())
        ).defect.structure.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped {type(self).__name__} for bulk composition {formula}, with "
            f"{len(self.defect_entries)} defect entries in self.defect_entries. Available "
            f"attributes:\n{properties}\n\nAvailable methods:\n{methods}"
        )

    def __getattr__(self, attr):
        """
        Redirects an unknown attribute/method call to the ``defect_sets``
        dictionary attribute, if the attribute doesn't exist in this class.
        """
        # Return the attribute if it exists in self.__dict__
        if attr in self.__dict__:
            return self.__dict__[attr]

        # Check if the attribute exists in defect_sets:
        if hasattr(self.defect_sets, attr):
            return getattr(self.defect_sets, attr)

        # If all else fails, raise an AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __getitem__(self, key):
        """
        Makes this object subscriptable, so that it can be indexed like a
        dictionary, using the ``defect_sets`` dictionary attribute.
        """
        return self.defect_sets[key]

    def __setitem__(self, key, value):
        """
        Set the value of a specific key (defect name) in the ``defect_sets``
        dictionary.
        """
        self.defect_sets[key] = value

    def __delitem__(self, key):
        """
        Deletes the specified input set from the ``defect_sets`` dictionary.
        """
        del self.defect_sets[key]

    def __contains__(self, key):
        """
        Returns True if the ``defect_sets`` dictionary contains the specified
        defect name.
        """
        return key in self.defect_sets

    def __len__(self):
        r"""
        Returns the number of input sets in the ``defect_sets`` dictionary.
        """
        return len(self.defect_sets)

    def __iter__(self):
        """
        Returns an iterator over the ``defect_sets`` dictionary.
        """
        return iter(self.defect_sets)
