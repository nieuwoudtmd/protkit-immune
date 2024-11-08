#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Authors: Mechiel Nieuwoudt (MN)
# Contact: mechiel@silicogenesis.com
# License: GPLv3

"""
Implements class `Alignment` for aligning protein structures.

This module utilizes the ProDy library to compute transformations
that minimize the RMSD between two sets of protein coordinates.
"""

from prody import calcTransformation, calcRMSD
import numpy as np
from typing import Tuple, List

class Alignment:
    def __init__(self, mobile_coords: List[Tuple[float, float, float]], target_coords: List[Tuple[float, float, float]]):
        """
        Constructor for Alignment. Prepares mobile and target coordinates for alignment.

        Args:
            mobile_coords (List[Tuple[float, float, float]]): List of coordinates to align (mobile structure).
            target_coords (List[Tuple[float, float, float]]): List of coordinates to align to (target structure).
        """
        self._mobile_coords = np.array(mobile_coords)
        self._target_coords = np.array(target_coords)
        self._transformation = None
        self._rmsd = None

    @staticmethod
    def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
        """
        Calculates the RMSD between two sets of coordinates.

        Args:
            coords1 (np.ndarray): First set of coordinates.
            coords2 (np.ndarray): Second set of coordinates to compare against.

        Returns:
            float: The calculated RMSD.
        """
        return calcRMSD(coords1, coords2)

    def align(self) -> Tuple[float, 'Transformation']:
        """
        Aligns the mobile coordinates to the target coordinates using ProDy's transformation.

        Returns:
            Tuple[float, Transformation]: RMSD after alignment and the transformation object.
        """
        # Compute transformation to minimize RMSD
        self._transformation = calcTransformation(self._mobile_coords, self._target_coords)
        transformed_mobile = self._transformation.apply(self._mobile_coords)

        # Calculate RMSD between transformed mobile and target using the static method
        self._rmsd = self.calculate_rmsd(transformed_mobile, self._target_coords)

        print(f"RMSD after alignment: {self._rmsd:.3f} Å")
        return self._rmsd, self._transformation

    @property
    def rmsd(self) -> float:
        """
        Getter for the last calculated RMSD.

        Returns:
            float: The RMSD from the latest alignment.
        """
        if self._rmsd is None:
            raise ValueError("RMSD not available. Run `align()` first.")
        return self._rmsd

    @property
    def mobile_coords(self) -> np.ndarray:
        """
        Getter for the mobile coordinates.

        Returns:
            np.ndarray: The mobile coordinates.
        """
        return self._mobile_coords

    @property
    def target_coords(self) -> np.ndarray:
        """
        Getter for the target coordinates.

        Returns:
            np.ndarray: The target coordinates.
        """
        return self._target_coords

    @property
    def transformation(self):
        """
        Getter for the transformation.

        Returns:
            Transformation: The transformation applied during alignment.
        """
        if self._transformation is None:
            raise ValueError("No transformation has been applied. Run `align()` first.")
        return self._transformation
