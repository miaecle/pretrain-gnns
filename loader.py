import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import torch_data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain

assert "DATASET_ROOT" in os.environ, "Environment variable DATASET_ROOT must be set"

# %% Atom and bond features
# Allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


num_atom_feautres = 2
def get_atom_feature(atom):
    feat = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
    return feat


num_bond_features = 2
def get_bond_feature(bond):
    feat = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
    return feat


# %% Graph featurization: rdkit <=> tg.data <=> nx
def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = get_atom_feature(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = get_bond_feature(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def graph_data_obj_to_mol_simple(d):
    """
    Convert pytorch geometric data obj to rdkit mol object. NB: Uses simplified
    atom and bond features, and represent as indices.
    :param: data_x:
    :param: data_edge_index:
    :param: data_edge_attr
    :return:
    """
    data_x = d.x.cpu().numpy()
    data_edge_index = d.edge_index.cpu().numpy()
    data_edge_attr = d.edge_attr.cpu().numpy()
    
    mol = Chem.RWMol()

    # atoms
    atom_features = data_x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i][:2]
        atomic_num = allowable_features['possible_atomic_num_list'][atomic_num_idx]
        chirality_tag = allowable_features['possible_chirality_list'][chirality_tag_idx]
        atom = Chem.Atom(atomic_num)
        atom.SetChiralTag(chirality_tag)
        mol.AddAtom(atom)

    # bonds
    edge_index = data_edge_index.cpu().numpy()
    edge_attr = data_edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j][:2]
        bond_type = allowable_features['possible_bonds'][bond_type_idx]
        bond_dir = allowable_features['possible_bond_dirs'][bond_dir_idx]
        mol.AddBond(begin_idx, end_idx, bond_type)
        # set bond direction
        new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        new_bond.SetBondDir(bond_dir)

    # Chem.SanitizeMol(mol) # fails for COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(
    # C)C(OC)=C2C)C=C1, when aromatic bond is possible
    # when we do not have aromatic bonds
    # Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    return mol


def graph_data_obj_to_nx_simple(d):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # atoms
    atom_features = d.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i][:2]
        G.add_node(i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx)
        pass

    # bonds
    edge_index = d.edge_index.cpu().numpy()
    edge_attr = d.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j][:2]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx,
                       bond_dir_idx=bond_dir_idx)

    return G


def nx_to_graph_data_obj_simple(G):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # atoms
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['atom_num_idx'], node['chirality_tag_idx']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['bond_type_idx'], edge['bond_dir_idx']]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# %% Miscellaneous functions
def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def create_standardized_mol_id(smiles):
    """
    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol != None:
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return


def to_list(x):
    if not isinstance(x, (tuple, list)):
        x = [x]
    return x


# %% Main dataset class
class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])


    def get(self, idx):
        d = Data()
        for key in self.d.keys:
            item, slices = self.d[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[d.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            d[key] = item[s]
        return d


    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')


    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list


    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'


    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        files = to_list(self.raw_file_names)
        return [os.path.join(self.raw_dir, f) for f in files]


    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        files = to_list(self.processed_file_names)
        return [os.path.join(self.processed_dir, f) for f in files]


    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')


    def batch_import_from_rdkit(self, rdkit_mol_objs, ids=None, labels=None):
        if ids is None:
            ids = np.arange(len(rdkit_mol_objs))
        assert len(rdkit_mol_objs) == len(ids)
        
        data_list = []
        for i in range(len(ids)):
            d = mol_to_graph_data_obj_simple(rdkit_mol_objs[i])
            d.id = torch.tensor([ids[i]])
            if labels is not None:
                d.y = torch.tensor(labels[i, :])
            data_list.append(d)
        return data_list


    def process(self):
        smiles_list = []
        data_list = []

        if self.dataset == 'zinc_standard_agent':
            smiles_list, rdkit_mol_objs, _, ids = _load_zinc_dataset(self.raw_paths[0])
            data_list = self.batch_import_from_rdkit(rdkit_mol_objs, ids=ids, labels=None)

        elif self.dataset == 'chembl_filtered':
            smiles_list, rdkit_mol_objs, labels, folds = _load_chembl_with_labels_dataset(self.raw_dir)
            valid_mol_idx = _filter_mols(smiles_list, 
                                         rdkit_mol_objs, 
                                         downstream=True, 
                                         test_only=True, 
                                         molecular_weight=True)
            smiles_list = [smiles_list[i] for i in valid_mol_idx]
            rdkit_mol_objs = [rdkit_mol_objs[i] for i in valid_mol_idx]
            labels = labels[np.array(valid_mol_idx)]
            folds = folds[np.array(valid_mol_idx)]

            data_list = self.batch_import_from_rdkit(rdkit_mol_objs, ids=None, labels=labels)
            for i, d in enumerate(data_list):
                d.fold = folds[i]

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = _load_bbbp_dataset(self.raw_paths[0])
            data_list = self.batch_import_from_rdkit(rdkit_mol_objs, ids=None, labels=labels)
        
        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = _load_clintox_dataset(self.raw_paths[0])
            data_list = self.batch_import_from_rdkit(rdkit_mol_objs, ids=None, labels=labels)

        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = _load_hiv_dataset(self.raw_paths[0])
            data_list = self.batch_import_from_rdkit(rdkit_mol_objs, ids=None, labels=labels)

        elif self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = _load_tox21_dataset(self.raw_paths[0])
            data_list = self.batch_import_from_rdkit(rdkit_mol_objs, ids=None, labels=labels)

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = _load_esol_dataset(self.raw_paths[0])
            data_list = self.batch_import_from_rdkit(rdkit_mol_objs, ids=None, labels=labels)

        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = _load_freesolv_dataset(self.raw_paths[0])
            data_list = self.batch_import_from_rdkit(rdkit_mol_objs, ids=None, labels=labels)

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = _load_lipophilicity_dataset(self.raw_paths[0])
            data_list = self.batch_import_from_rdkit(rdkit_mol_objs, ids=None, labels=labels)

        # elif self.dataset == 'pcba':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_pcba_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         print(i)
        #         rdkit_mol = rdkit_mol_objs[i]
        #         # # convert aromatic bonds to double bonds
        #         # Chem.SanitizeMol(rdkit_mol,
        #         #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #         data = mol_to_graph_data_obj_simple(rdkit_mol)
        #         # manually add mol id
        #         data.id = torch.tensor(
        #             [i])  # id here is the index of the mol in
        #         # the dataset
        #         data.y = torch.tensor(labels[i, :])
        #         data_list.append(data)
        #         data_smiles_list.append(smiles_list[i])

        # elif self.dataset == 'pcba_pretrain':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_pcba_dataset(self.raw_paths[0])
        #     downstream_inchi = set(pd.read_csv(os.path.join(self.root,
        #                                                     'downstream_mol_inchi_may_24_2019'),
        #                                        sep=',', header=None)[0])
        #     for i in range(len(smiles_list)):
        #         print(i)
        #         if '.' not in smiles_list[i]:   # remove examples with
        #             # multiples species
        #             rdkit_mol = rdkit_mol_objs[i]
        #             mw = Descriptors.MolWt(rdkit_mol)
        #             if 50 <= mw <= 900:
        #                 inchi = create_standardized_mol_id(smiles_list[i])
        #                 if inchi != None and inchi not in downstream_inchi:
        #                     # # convert aromatic bonds to double bonds
        #                     # Chem.SanitizeMol(rdkit_mol,
        #                     #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #                     data = mol_to_graph_data_obj_simple(rdkit_mol)
        #                     # manually add mol id
        #                     data.id = torch.tensor(
        #                         [i])  # id here is the index of the mol in
        #                     # the dataset
        #                     data.y = torch.tensor(labels[i, :])
        #                     data_list.append(data)
        #                     data_smiles_list.append(smiles_list[i])

        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, 'smiles.csv'), 
                                  index=False, 
                                  header=False)

        collated_data, slices = self.collate(data_list)
        torch.save((collated_data, slices), self.processed_paths[0])


# def merge_dataset_objs(dataset_1, dataset_2):
#     """
#     Naively merge 2 molecule dataset objects, and ignore identities of
#     molecules. Assumes both datasets have multiple y labels, and will pad
#     accordingly. ie if dataset_1 has obj_1 with y dim 1310 and dataset_2 has
#     obj_2 with y dim 128, then the resulting obj_1 and obj_2 will have dim
#     1438, where obj_1 have the last 128 cols with 0, and obj_2 have
#     the first 1310 cols with 0.
#     :return: pytorch geometric dataset obj, with the x, edge_attr, edge_index,
#     new y attributes only
#     """
#     d_1_y_dim = dataset_1[0].y.size()[0]
#     d_2_y_dim = dataset_2[0].y.size()[0]

#     data_list = []
#     # keep only x, edge_attr, edge_index, padded_y then append
#     for d in dataset_1:
#         old_y = d.y
#         new_y = torch.cat([old_y, torch.zeros(d_2_y_dim, dtype=torch.long)])
#         data_list.append(Data(x=d.x, edge_index=d.edge_index,
#                               edge_attr=d.edge_attr, y=new_y))

#     for d in dataset_2:
#         old_y = d.y
#         new_y = torch.cat([torch.zeros(d_1_y_dim, dtype=torch.long), old_y.long()])
#         data_list.append(Data(x=d.x, edge_index=d.edge_index,
#                               edge_attr=d.edge_attr, y=new_y))

#     # create 'empty' dataset obj. Just randomly pick a dataset and root path
#     # that has already been processed
#     new_dataset = MoleculeDataset(root='dataset/chembl_with_labels',
#                                   dataset='chembl_with_labels', empty=True)
#     # collate manually
#     new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)

#     return new_dataset


# %% Data loading functions
def _load_zinc_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, None, list of mol ids
    """
    input_df = pd.read_csv(input_path, sep=',', compression='gzip', dtype='str')
    smiles_list = list(input_df['smiles'])
    zinc_id_list = list(input_df['zinc_id'])
    
    labels = None
    valid_smiles_list = []
    rdkit_mol_objs_list = []
    ids = []
    for id, smi in zip(zinc_id_list, smiles_list):
        try:
            rdkit_mol = AllChem.MolFromSmiles(smi)
            if not rdkit_mol is None:  # ignore invalid mol objects
                id = int(id.split('ZINC')[1].lstrip('0'))
                rdkit_mol_objs_list.append(rdkit_mol)
                valid_smiles_list.append(smi)
                ids.append(id)                
        except:
            continue
    return valid_smiles_list, rdkit_mol_objs_list, labels, ids


def _load_chembl_with_labels_dataset(raw_folder):
    """
    Data from 'Large-scale comparison of machine learning methods for drug target prediction on ChEMBL'
    :param raw_path: path to the folder containing the reduced chembl dataset
    :return: list of smiles, list of preprocessed rdkit mol obj, np.array 
    containing the labels, np.array of fold indices
    """

    # 1. load folds and labels
    with open(os.path.join(raw_folder, 'labelsHard.pckl'), 'rb') as f:
        targetMat = pickle.load(f)
        sampleAnnInd = pickle.load(f)
        assert list(sampleAnnInd) == list(range(len(sampleAnnInd)))
        targetMat = targetMat.copy().tocsr()
        targetMat.sort_indices()

    with open(os.path.join(raw_folder, 'folds0.pckl'), 'rb') as f:
        folds = pickle.load(f)
        fold_idx = -np.ones((len(sampleAnnInd),))
        for i, fold in enumerate(folds):
            fold_idx[np.array(fold)] = i
        assert fold_idx.min() >= 0

    denseLabels = targetMat.A # possible values are {-1, 0, 1}

    # 2. load structures
    with open(os.path.join(raw_folder, 'chembl20LSTM.pckl'), 'rb') as f:
        rdkitArr = pickle.load(f)
        assert len(rdkitArr) == denseLabels.shape[0]

    valid_mol_idx = []
    mol_list = []
    smiles = []
    for i, mol in enumerate(rdkitArr):
        if not mol is None:
            mol_species_list = split_rdkit_mol_obj(mol)
            if len(mol_species_list) > 0:
                largest_mol = get_largest_mol(mol_species_list)
                if mol.GetNumAtoms() > 2:
                    valid_mol_idx.append(i)
                    mol_list.append(largest_mol)
                    smiles.append(AllChem.MolToSmiles(largest_mol))
    
    denseLabels = denseLabels[np.array(valid_mol_idx)]
    fold_idx = fold_idx[np.array(valid_mol_idx)]
    return smiles, mol_list, denseLabels, fold_idx


def _load_bbbp_dataset(input_path, remove_invalid_mols=True):
    """
    :param input_path:
    :param remove_invalid_mols:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['p_np']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    labels = labels.values
    # there are no nans

    # Mask invalid molecules
    invalid_mol_idx = [i for i, mol in enumerate(rdkit_mol_objs_list) if mol is None]
    smiles_list = [smi if not i in invalid_mol_idx else None for i, smi in enumerate(smiles_list)]
    if remove_invalid_mols:
        valid_mol_idx = sorted(set(range(len(smiles_list))) - set(invalid_mol_idx))
        smiles_list = [smiles_list[i] for i in valid_mol_idx]
        rdkit_mol_objs_list = [rdkit_mol_objs_list[i] for i in valid_mol_idx]
        labels = labels[np.array(valid_mol_idx)]
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _load_clintox_dataset(input_path, remove_invalid_mols=True):
    """
    :param input_path:
    :param remove_invalid_mols:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    labels = labels.values
    # there are no nans
    
    # Mask invalid molecules
    invalid_mol_idx = [i for i, mol in enumerate(rdkit_mol_objs_list) if mol is None]
    smiles_list = [smi if not i in invalid_mol_idx else None for i, smi in enumerate(smiles_list)]
    if remove_invalid_mols:
        valid_mol_idx = sorted(set(range(len(smiles_list))) - set(invalid_mol_idx))
        smiles_list = [smiles_list[i] for i in valid_mol_idx]
        rdkit_mol_objs_list = [rdkit_mol_objs_list[i] for i in valid_mol_idx]
        labels = labels[np.array(valid_mol_idx)]
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _load_hiv_dataset(input_path, remove_invalid_mols=False):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    labels = labels.values
    # there are no nans
    # No invalid molecules in esol
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _load_tox21_dataset(input_path, remove_invalid_mols=False):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    labels = labels.values
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _load_esol_dataset(input_path, remove_invalid_mols=False):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    labels = labels.values
    # No invalid molecules in esol
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _load_freesolv_dataset(input_path, remove_invalid_mols=False):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['expt']
    labels = labels.values
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _load_lipophilicity_dataset(input_path, remove_invalid_mols=False):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    labels = labels.values
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _filter_mols(smis, mols, downstream=True, test_only=True, molecular_weight=True):
    assert len(smis) == len(mols)
    valid_idx = [i for i, mol in enumerate(mols) if not mol is None]
    
    if downstream: # Filter for downstream task molecules
        from splitters import scaffold_split
        downstream_datasets = [
            'bbbp',
            'clintox',
            'esol',
            'freesolv',
            'hiv',
            'lipophilicity',
            'tox21',
            ]
        downstream_inchi_set = set()
        for d in downstream_datasets:
            print(d)
            d_path = os.path.join(os.environ["DATASET_ROOT"], d)
            dataset = MoleculeDataset(d_path, dataset=d)            
            smis = pd.read_csv(os.path.join(\
                dataset.processed_dir, 'smiles.csv'), header=None)[0].tolist()
            _, _, _, (train_smiles, valid_smiles, test_smiles) = \
                scaffold_split(dataset, smis, task_idx=None, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, return_smiles=True)

            excluded_smiles = test_smiles + valid_smiles
            if not test_only:
                excluded_smiles += train_smiles

            for smiles in excluded_smiles:
                species_list = smiles.split('.')
                for s in species_list:  
                    inchi = create_standardized_mol_id(s)
                    downstream_inchi_set.add(inchi)
        valid_idx = [i for i in valid_idx if \
            create_standardized_mol_id(smis[i]) not in downstream_inchi_set]
    
    if molecular_weight: # Filter for molecular weight window [50, 900]
        valid_idx = [i for i in valid_idx if \
            50 <= Descriptors.MolWt(mols[i]) <= 900]
    return valid_idx


# %% Main
def create_all_datasets():
    downstream_dir = [
            'bbbp',
            'clintox',
            'esol',
            'freesolv',
            'hiv',
            'lipophilicity',
            'tox21',
            "zinc_standard_agent",
            # "chembl_filtered",
            ]
    for dataset_name in downstream_dir:
        print(dataset_name)
        root = os.path.join(os.environ["DATASET_ROOT"], dataset_name)
        os.makedirs(os.path.join(root, "processed"), exist_ok=True)
        dataset = MoleculeDataset(root, dataset=dataset_name)
        print(dataset)
    return


if __name__ == "__main__":
    create_all_datasets()

