# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from protenix.data.constants import ELEMS, STD_RESIDUES


class Token(object):
    """
    Used to store information related to Tokens.

    Example:
    >>> token = Token(1)
    >>> token.value
    1
    >>> token.atom_indices = [1, 2, 3]
    """

    def __init__(self, value, **kwargs):
        self.value = value
        self._annot = {}
        for name, annotation in kwargs.items():
            self._annot[name] = annotation

    def __getattr__(self, attr):
        if attr in super().__getattribute__("_annot"):
            return self._annot[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )

    def __repr__(self):
        annot_lst = []
        for k, v in self._annot.items():
            annot_lst.append(f"{k}={v}")
        return f'Token({self.value}, {",".join(annot_lst)})'

    def __setattr__(self, attr, value):
        if attr == "_annot":
            super().__setattr__(attr, value)
        elif attr == "value":
            super().__setattr__(attr, value)
        else:
            self._annot[attr] = value


class TokenArray(object):
    """
    A group of Token objects used for batch operations.
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens

    def __repr__(self):
        repr_str = "TokenArray(\n"
        for token in self.tokens:
            repr_str += f"\t{token}\n"
        repr_str += ")"
        return repr_str

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        for token in self.tokens:
            yield token

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.tokens[index]
        else:
            return TokenArray([self.tokens[i] for i in index])

    def get_annotation(self, category):
        return [token._annot[category] for token in self.tokens]

    def set_annotation(self, category, values):
        assert len(values) == len(
            self.tokens
        ), "Length of values must match the number of tokens"
        for token, value in zip(self.tokens, values):
            token._annot[category] = value

    def get_values(self):
        return [token.value for token in self.tokens]


class AtomArrayTokenizer(object):
    """
    Tokenize an AtomArray object into a list of Token object.
    """

    def __init__(self, atom_array: AtomArray):
        self.atom_array = atom_array

    def tokenize(self) -> list[Token]:
        """
        Ref: AlphaFold3 SI Chapter 2.6
        Tokenize an AtomArray object into a list of Token object.

        Returns:
           list : a list of Token object.
        """
        atom_count = len(self.atom_array)
        if atom_count == 0:
            return []

        # ``residue_iter()`` creates an AtomArray slice for every residue and
        # indexing such a slice creates an Atom object for every atom. Work
        # directly on the annotation arrays using the same residue boundaries.
        residue_starts = struc.get_residue_starts(
            self.atom_array,
            add_exclusive_stop=True,
        )
        atom_names = self.atom_array.atom_name
        elements = self.atom_array.element
        residue_names = self.atom_array.res_name[residue_starts[:-1]]
        molecule_types = self.atom_array.mol_type[residue_starts[:-1]]

        tokens = []
        for start, stop, res_name, mol_type in zip(
            residue_starts[:-1],
            residue_starts[1:],
            residue_names,
            molecule_types,
        ):
            res_token = STD_RESIDUES.get(res_name, None)
            if res_token is not None and mol_type != "ligand":
                # for std residues
                token = Token(res_token)
                token.atom_indices = list(range(start, stop))
                token.atom_names = atom_names[start:stop].tolist()
                tokens.append(token)
            else:
                # for ligand and non-std residues
                for atom_index in range(start, stop):
                    atom_elem = elements[atom_index]
                    atom_token = ELEMS.get(atom_elem, None)
                    if atom_token is None:
                        raise ValueError(f"Unknown atom element: {atom_elem}")
                    token = Token(atom_token)
                    token.atom_indices = [atom_index]
                    token.atom_names = [atom_names[atom_index]]
                    tokens.append(token)

        return tokens

    def _set_token_annotations(self, token_array: TokenArray) -> TokenArray:
        """
        Set annotations for the token_array.

        The annotations include:
            - centre_atom_index: the atom indices of the token in the atom array

        Args:
            token_array (TokenArray): TokenArray object created by tokenize bioassembly AtomArray.

        Returns:
            TokenArray: TokenArray object with annotations.
        """
        centre_atom_indices = np.where(self.atom_array.centre_atom_mask == 1)[0]
        token_array.set_annotation("centre_atom_index", centre_atom_indices)
        assert len(token_array) == len(centre_atom_indices)
        return token_array

    def get_token_array(self) -> TokenArray:
        """
        Get TokenArray object with annotations (atom_indices, centre_atom_index).

        Returns:
            TokenArray: The TokenArray object with annotations.
                TokenArray(
                Token(1, atom_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],centre_atom_index=2,
                    atom_names=['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'])
                Token(15, atom_indices=[11, 12, 13, 14, 15, 16],centre_atom_index=13,
                    atom_names=['N', 'CA', 'C', 'O', 'CB', 'OG'])
                Token(15, atom_indices=[17, 18, 19, 20, 21, 22],centre_atom_index=19,
                    atom_names=['N', 'CA', 'C', 'O', 'CB', 'OG'])
                    )
                it satisfy the following format
                Token($token_index,  atom_indices=[global_atom_indexs],
                    centre_atom_index=global_atom_indexs,atom_names=[names])
        """
        tokens = self.tokenize()
        token_array = TokenArray(tokens=tokens)
        token_array = self._set_token_annotations(token_array=token_array)
        return token_array
