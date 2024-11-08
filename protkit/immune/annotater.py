#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Authors:  Mechiel Nieuwoudt (MN)
# Contact:  mechiel@silicogenesis.com
# License:  GPLv3

"""
Implements class `Annotator` to annotate proteins, chains, and residues
with numbering and region information using the abnumber tool.
"""

from typing import List, Union
from abnumber import Chain as AbChain, ChainParseError
from abnumber.chain import MultipleDomainsChainParseError

from protkit.structure.residue import Residue
from protkit.structure.chain import Chain
from protkit.structure.protein import Protein


class Annotator:
    """
    Class to annotate proteins, chains, and residues with numbering and region information.
    """

    @staticmethod
    def annotate_residues(chain: Chain, ab_chain: AbChain, assign_attributes: bool = True) -> None:
        """
        Annotate the residues in a chain with numbering and region information.

        Args:
            chain (Chain): The chain to annotate.
            ab_chain (AbChain): The abnumber Chain object already created for this chain.
            assign_attributes (bool): Whether to assign the annotations to the residues.
        """
        # Map abnumber positions to residues
        ab_positions = list(ab_chain.positions.items())
        residues = list(chain.residues)

        for residue, (pos, aa) in zip(residues, ab_positions):
            if assign_attributes:
                residue.set_attribute("scheme_region", pos.get_region())
                residue.set_attribute("scheme_number", pos.format(chain_type=False))

    @staticmethod
    def annotate_chain(chain: Chain,
                       scheme: str = 'imgt',
                       cdr_definition: Union[str, None] = None,
                       assign_attributes: bool = True) -> Chain:
        """
        Annotate a chain with numbering and region information.

        Args:
            chain (Chain): The chain to annotate.
            scheme (str): The numbering scheme to use.
            cdr_definition (str): The scheme for defining CDR regions.
            assign_attributes (bool): Whether to assign the annotations to the chain and its residues.

        Returns:
            Chain: The annotated chain.
        """
        try:
            ab_chain = AbChain(
                sequence=chain.sequence,
                scheme=scheme,
                cdr_definition=cdr_definition,
                assign_germline=True
            )
        except (ChainParseError, MultipleDomainsChainParseError) as e:
            raise ValueError(f"Error parsing sequence: {e}")

        # Annotate residues using the existing ab_chain object
        Annotator.annotate_residues(chain, ab_chain=ab_chain, assign_attributes=assign_attributes)

        # Set chain-level attributes
        if assign_attributes:
            chain.set_attribute("chain_type", ab_chain.chain_type)
            chain.set_attribute("species", ab_chain.species)
            chain.set_attribute("v_gene", ab_chain.v_gene)
            chain.set_attribute("j_gene", ab_chain.j_gene)

        return chain

    @staticmethod
    def annotate_protein(protein: Protein,
                         scheme: str = 'imgt',
                         cdr_definition: Union[str, None] = None,
                         assign_attributes: bool = True) -> Protein:
        """
        Annotate a protein with numbering and region information.

        Args:
            protein (Protein): The protein to annotate.
            scheme (str): The numbering scheme to use.
            cdr_definition (str): The scheme for defining CDR regions.
            assign_attributes (bool): Whether to assign the annotations to the protein and its chains and residues.

        Returns:
            Protein: The annotated protein.
        """
        for chain in protein.chains:
            Annotator.annotate_chain(chain, scheme=scheme, cdr_definition=cdr_definition, assign_attributes=assign_attributes)
        return protein
