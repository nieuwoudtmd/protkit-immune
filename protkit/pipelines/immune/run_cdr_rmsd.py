#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Authors: Mechiel Nieuwoudt (MN)
# Contact: mechiel@silicogenesis.com
# License: GPLv3

"""
Script to calculate the antibody/nanobody CDR RMSD between a predicted protein structure and an experimental structure,
aligning specified chains together and ensuring correct matching of heavy and light chain atoms.
"""

from protkit.file_io import PDBIO
from protkit.immune import Annotator
from protkit.geometry.alignment import Alignment
import numpy as np

# Helper function to extract backbone coordinates
def get_backbone_coordinates(chain, regions=None, atom_types=["CA"]):
    """
    Extract backbone atom coordinates from a chain based on region filtering.
    Args:
        chain (Chain): The chain from which to extract coordinates.
        regions (List[str]): List of regions to include (e.g., ['Framework', 'CDR1']).
        atom_types (List[str]): List of atom types to include (e.g., ['CA']).
    Returns:
        List[List[float]]: List of [x, y, z] coordinates.
    """
    # Get residues matching the specified regions
    if regions:
        filtered_residues = [res for res in chain.residues if res.get_attribute("scheme_region") in regions]
    else:
        filtered_residues = chain.residues

    # From the filtered residues, get atoms matching the atom types
    backbone_atoms = []
    for residue in filtered_residues:
        for atom in residue.atoms:
            if atom.atom_type in atom_types:
                backbone_atoms.append(atom)

    # Return coordinates as a list of [x, y, z]
    return [[atom.x, atom.y, atom.z] for atom in backbone_atoms]

# Main function to run the pipeline
def run_cdr_rmsd_pipeline(predicted_pdb_file, experimental_pdb_file, predicted_chain_ids, experimental_chain_ids, numbering_scheme='chothia', atom_types=['CA']):
    """
    Runs the pipeline to calculate CDR RMSD between predicted and experimental structures.
    Alignment is done on all specified chains together, ensuring correct matching of chain types.

    Args:
        predicted_pdb_file (str): Path to the predicted PDB file.
        experimental_pdb_file (str): Path to the experimental PDB file.
        predicted_chain_ids (List[str]): List of chain IDs to keep for the predicted structure.
        experimental_chain_ids (List[str]): List of chain IDs to keep for the experimental structure.
        numbering_scheme (str): Numbering scheme to use for annotation.
        atom_types (List[str]): List of atom types to use for coordinate extraction.

    Returns:
        dict: Dictionary containing RMSD results for each chain and CDR.
    """
    # Load predicted and experimental structures
    predicted_protein = PDBIO.load(predicted_pdb_file)[0]
    experimental_protein = PDBIO.load(experimental_pdb_file)[0]

    # Keep only the specified chains in each structure
    predicted_protein.keep_chains(predicted_chain_ids)
    experimental_protein.keep_chains(experimental_chain_ids)

    # Remove hetero residues
    predicted_protein.remove_hetero_residues()
    experimental_protein.remove_hetero_residues()

    # Annotate all chains in both structures
    for chain in predicted_protein.chains:
        Annotator.annotate_chain(chain, scheme=numbering_scheme, assign_attributes=True)

    for chain in experimental_protein.chains:
        Annotator.annotate_chain(chain, scheme=numbering_scheme, assign_attributes=True)

    # Build mappings from chain_type to list of chains
    predicted_chains_by_type = {}
    for chain in predicted_protein.chains:
        chain_type = getattr(chain, '_chain_type', None)
        if chain_type:
            predicted_chains_by_type.setdefault(chain_type, []).append(chain)
    experimental_chains_by_type = {}
    for chain in experimental_protein.chains:
        chain_type = getattr(chain, '_chain_type', None)
        if chain_type:
            experimental_chains_by_type.setdefault(chain_type, []).append(chain)

    # Collect framework coordinates from all specified chains, ensuring correct matching
    predicted_framework_coords = []
    experimental_framework_coords = []
    chain_mappings = []

    # Ensure that heavy chain atoms are aligned with heavy chain atoms
    for chain_type in predicted_chains_by_type:
        if chain_type not in experimental_chains_by_type:
            print(f"Chain type {chain_type} present in predicted structure but not in experimental structure.")
            continue

        pred_chains = predicted_chains_by_type[chain_type]
        exp_chains = experimental_chains_by_type[chain_type]

        if len(pred_chains) != len(exp_chains):
            print(f"Number of {chain_type} chains does not match between predicted and experimental structures.")
            continue

        # Pair predicted and experimental chains of the same type
        for pred_chain, exp_chain in zip(pred_chains, exp_chains):
            # Collect framework coordinates
            pred_coords = get_backbone_coordinates(pred_chain, regions=["FR1", "FR2", "FR3", "FR4"], atom_types=atom_types)
            exp_coords = get_backbone_coordinates(exp_chain, regions=["FR1", "FR2", "FR3", "FR4"], atom_types=atom_types)

            if len(pred_coords) != len(exp_coords):
                print(f"Number of atoms in predicted and experimental framework regions do not match for chain {pred_chain.id} ({chain_type}).")
                continue

            # Append to the overall coordinate lists
            predicted_framework_coords.extend(pred_coords)
            experimental_framework_coords.extend(exp_coords)

            # Keep track of chain mapping
            chain_mappings.append((chain_type, pred_chain, exp_chain))

    # Convert to numpy arrays
    predicted_framework_coords = np.array(predicted_framework_coords)
    experimental_framework_coords = np.array(experimental_framework_coords)

    # Check that the number of atoms is the same
    if predicted_framework_coords.shape != experimental_framework_coords.shape or predicted_framework_coords.size == 0:
        print(f"Total number of atoms in predicted and experimental framework regions do not match or no atoms found.")
        return {}

    # Perform superimposition and calculate RMSD on the framework regions using imported Alignment class
    alignment = Alignment(predicted_framework_coords, experimental_framework_coords)
    rmsd_framework, framework_transformation = alignment.align()

    # Store the overall framework RMSD
    results = {'framework_rmsd': rmsd_framework, 'cdr_rmsd': {}}

    # Now, calculate CDR RMSD for each chain
    for chain_type, pred_chain, exp_chain in chain_mappings:
        chain_id = f"{chain_type}_{pred_chain.id}"

        results['cdr_rmsd'][chain_id] = {}

        # For each CDR region
        for cdr in ["CDR1", "CDR2", "CDR3"]:
            # Extract backbone coordinates for the CDR region
            predicted_cdr_coords = get_backbone_coordinates(pred_chain, regions=[cdr], atom_types=atom_types)
            experimental_cdr_coords = get_backbone_coordinates(exp_chain, regions=[cdr], atom_types=atom_types)

            # Convert to NumPy arrays
            predicted_cdr_coords = np.array(predicted_cdr_coords)
            experimental_cdr_coords = np.array(experimental_cdr_coords)

            # Check that the number of atoms is the same
            if predicted_cdr_coords.shape != experimental_cdr_coords.shape or predicted_cdr_coords.size == 0:
                print(f"Number of atoms in predicted and experimental {cdr} do not match for chain {chain_id} or no atoms found.")
                continue

            # Apply the transformation based on framework to the predicted CDR coordinates
            transformed_predicted_cdr_coords = framework_transformation.apply(predicted_cdr_coords)

            # Calculate RMSD between transformed predicted and experimental CDR coordinates
            cdr_rmsd = alignment.calculate_rmsd(transformed_predicted_cdr_coords, experimental_cdr_coords)

            # Store the CDR RMSD
            results['cdr_rmsd'][chain_id][cdr] = cdr_rmsd

    # Return the results
    return results