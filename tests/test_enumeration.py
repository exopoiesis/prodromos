"""Tests для multi_endpoint_enumeration.py."""
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from prodromos.multi_endpoint_enumeration import (
    mic_vec,
    remove_atom_and_add_H,
)


class TestMicVec:
    def test_zero_for_same_point(self):
        cell = np.eye(3) * 10
        v = mic_vec(np.zeros(3), np.zeros(3), cell)
        np.testing.assert_allclose(v, 0, atol=1e-10)

    def test_direct_vector(self):
        """Points within first cell: simple difference."""
        cell = np.eye(3) * 10
        v = mic_vec(np.array([0, 0, 0]), np.array([1, 0, 0]), cell)
        np.testing.assert_allclose(v, [1, 0, 0])

    def test_pbc_wrap(self):
        """Точки on opposite sides of cell wrap via MIC."""
        cell = np.eye(3) * 10
        # From (1, 0, 0) к (9, 0, 0): direct = 8, but MIC = -2
        v = mic_vec(np.array([1, 0, 0]), np.array([9, 0, 0]), cell)
        # MIC: difference should be -2 (shorter than +8)
        np.testing.assert_allclose(np.abs(v[0]), 2.0, atol=1e-10)


class TestRemoveAtomAndAddH:
    def test_atom_count_preserved(self):
        """Remove 1 atom + add H = same total atoms."""
        atoms = Atoms(
            symbols=["Fe", "Fe", "S", "S"],
            positions=np.eye(4)[:, :3] * 2,
            cell=[10, 10, 10],
            pbc=True,
        )
        h_pos = np.array([1.0, 1.0, 1.0])
        new = remove_atom_and_add_H(atoms, V_idx=0, h_position=h_pos)
        assert len(new) == len(atoms)

    def test_removes_correct_atom(self):
        """Specified atom should be missing in result."""
        atoms = Atoms(
            symbols=["Fe", "Fe", "S"],
            positions=np.array([[0, 0, 0], [5, 5, 5], [2, 2, 2]]),
            cell=[10, 10, 10],
            pbc=True,
        )
        h_pos = np.array([1.0, 1.0, 1.0])
        new = remove_atom_and_add_H(atoms, V_idx=0, h_position=h_pos)
        # Original Fe at (0,0,0) should be gone
        for pos in new.get_positions():
            assert not np.allclose(pos, [0, 0, 0])

    def test_h_at_specified_position(self):
        """H atom should be at h_position."""
        atoms = Atoms(
            symbols=["Fe", "S"],
            positions=np.array([[0, 0, 0], [3, 3, 3]]),
            cell=[10, 10, 10],
            pbc=True,
        )
        h_pos = np.array([1.5, 2.5, 3.5])
        new = remove_atom_and_add_H(atoms, V_idx=0, h_position=h_pos)
        # Find H in new structure
        h_idx_list = [i for i, s in enumerate(new.get_chemical_symbols()) if s == "H"]
        assert len(h_idx_list) == 1
        np.testing.assert_allclose(new.get_positions()[h_idx_list[0]], h_pos)

    def test_other_atoms_preserved(self):
        """Atoms не at V_idx должны сохранить positions."""
        atoms = Atoms(
            symbols=["Fe", "Fe", "S"],
            positions=np.array([[0, 0, 0], [5, 5, 5], [2, 2, 2]]),
            cell=[10, 10, 10],
            pbc=True,
        )
        h_pos = np.array([1.0, 1.0, 1.0])
        new = remove_atom_and_add_H(atoms, V_idx=0, h_position=h_pos)
        # Position (5,5,5) and (2,2,2) should still be there
        new_positions = new.get_positions()
        # First 2 atoms of new = atoms[1], atoms[2]; last = H
        np.testing.assert_allclose(new_positions[0], [5, 5, 5])
        np.testing.assert_allclose(new_positions[1], [2, 2, 2])
        np.testing.assert_allclose(new_positions[-1], h_pos)
