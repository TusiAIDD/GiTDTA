import random
from rdkit import Chem
import pickle
from torch_geometric.data import Data
from tqdm import trange
from Model import *
import pandas as pd


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])


def smiles2graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    c_size = molecule.GetNumAtoms()

    features = []
    for atom in molecule.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    x = torch.tensor(features, dtype=torch.float)

    edge_index = []
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))  # 添加反向边使图变为无向图
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, num_nodes=c_size)


def preEncodeData(fileName):
    data = pd.read_csv(f'data/GraphDTA/{fileName}_train.csv').dropna()

    graph_data_list = []
    for i in trange(len(data)):
        item = data.iloc[i]
        # esm_embedding = esm_data.iloc[i].values
        graph = smiles2graph(item['compound_iso_smiles'])
        if graph is not None:
            graph_data_list.append([graph, item['target_sequence'], item['affinity']])

    with open(f'data/{fileName}_training.pkl', 'wb') as f:
        pickle.dump(graph_data_list, f)


    data = pd.read_csv(f'data/GraphDTA/{fileName}_test.csv').dropna()

    graph_data_list = []
    for i in trange(len(data)):
        item = data.iloc[i]
        graph = smiles2graph(item['compound_iso_smiles'])
        if graph is not None:
            graph_data_list.append([graph, item['target_sequence'], item['affinity']])

    with open(f'data/{fileName}_test.pkl', 'wb') as f:
        pickle.dump(graph_data_list, f)


def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed_everything()

    params = {
        "fileName": "human",
        # "fileName": "celegans",
        # "fileName": "davis",
        # "fileName": "kiba",
        "BATCH_SIZE": 128,
        "EPOCH": 20,
        "loss": "CEP",
        "lr": 0.0001,
        "smile_input_dim": 78,
        "smiles_dim": 256,
        "e_layers": 8,
        "embed_dim": 20,
        "n_heads": 2,
        "target_dim": 128,
        "predictor_fc": 64,
        "dprate": 0.2,
        "lamda": 0,
        "seq_len": 1002,
        "output_attention": 128,
        "d_model": 8,
        "factor": 5,
        "d_ff": "None",
        "activation": "relu",
        "gin_layer_nums": 3,
        "max_AUC": 0
    }

    params['gin_layer_dims'] = [params['smiles_dim'] for _ in range(params['gin_layer_nums'])]

    fileName = params['fileName'] + str(params['seq_len'] - 2)
    model = UniModel(params)
    training_data = Sample(batch_size=params['BATCH_SIZE'],fileName=f'{fileName}_training' + '_float')
    test_data = Sample(batch_size=params['BATCH_SIZE'] * 10, fileName=f'{fileName}_test' + '_float')

    trainer = ModelTrainer(model=model, num_epochs=params['EPOCH'])
    trainer.test(test_data.dataloader, params=params)

