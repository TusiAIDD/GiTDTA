import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from tqdm.contrib import tenumerate
import torch.nn.functional as F
from iTransformer import Model
import torch.nn as nn
import torch


class LayerAttention(nn.Module):
    def __init__(self, num_layers):
        super(LayerAttention, self).__init__()
        self.num_layers = num_layers
        self.attention_weights = nn.Parameter(torch.randn(self.num_layers, 1))

    def forward(self, layer_representations):
        attention_scores = F.softmax(self.attention_weights, dim=0).view(1, self.num_layers, 1)
        weighted_representations = torch.sum(layer_representations * attention_scores, dim=1)
        return weighted_representations


class UniModel(nn.Module):
    def __init__(self, params):
        super(UniModel, self).__init__()
        self.gin_layers = nn.ModuleList()
        for i in range(len(params['gin_layer_dims'])):
            input_dim = params['smile_input_dim'] if i == 0 else params['gin_layer_dims'][i - 1]
            output_dim = params['gin_layer_dims'][i]
            self.gin_layers.append(GINConv(nn.Linear(input_dim, output_dim)))
        self.layer_attention = LayerAttention(len(params['gin_layer_dims']))
        if params['embed_dim'] != 20:
            self.embed_tokens = nn.Sequential(
                nn.Linear(20, params['embed_dim']),
                nn.ReLU(),
            )
        self.TargetModule = Model(params)
        self.seq_projector = nn.Sequential(
            nn.Linear(in_features=params['embed_dim'], out_features=params['target_dim']),
            nn.BatchNorm1d(params['target_dim']),
            nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_features=params['smiles_dim']+ params['target_dim'],
                      out_features=params['smiles_dim'] + params['target_dim']),
            nn.BatchNorm1d(params['smiles_dim'] + params['target_dim']),
            nn.ReLU(),

            nn.Linear(in_features=params['smiles_dim'] + params['target_dim'],
                      out_features=params['predictor_fc']),
            nn.BatchNorm1d(params['predictor_fc']),
            nn.ReLU(),

            nn.Linear(in_features=params['predictor_fc'],
                      out_features=1 if params['loss'] == 'MSE' else 2),
        )
        self.dropout = nn.Dropout(p=params['dprate'])

    def forward(self, smiles, seqs):
        try:
            seqs = self.dropout(self.embed_tokens(seqs))
        except:
            pass

        dec_out = self.TargetModule.forecast(seqs)
        target_representation = self.seq_projector(dec_out[:, -1:, :].view(dec_out.size()[0], dec_out.size()[2]))
        x, edge_index, smiles_batch = smiles.x, smiles.edge_index, smiles.batch
        layer_representations = []
        for gin_layer in self.gin_layers:
            x = gin_layer(x, edge_index)
            layer_representations.append(global_add_pool(x, smiles_batch))
        combined_representations = [torch.cat((layer_repr, target_representation), dim=1)
                                    for layer_repr in layer_representations]
        layer_representations_stack = torch.stack(combined_representations, dim=1)
        weighted_representation = self.layer_attention(layer_representations_stack)
        pred_affinity = self.predictor(weighted_representation)
        return pred_affinity


class ModelTrainer:
    def __init__(self, model, num_epochs, modelName=None):
        self.model = model.cuda()
        self.num_epochs = num_epochs
        self.modelName = modelName

    def rmse(self, y, f):
        rmse = np.sqrt(((y - f) ** 2).mean(axis=0))
        return rmse

    def mse(self, y, f):
        mse = ((y.astype(float).reshape(-1, 1) - f.astype(float).reshape(-1, 1)) ** 2).mean(axis=0)
        return mse[0]

    def test(self, test_loader, params=None):
        self.model.load_state_dict(torch.load(f"model/{params['fileName']}.pth"))
        self.model.eval()
        with torch.no_grad():

            all_labels = []
            all_predictions = []

            for i, (b_smiles, b_seqs, b_seqs_mask, b_affinity) in tenumerate(test_loader):
                b_smiles = b_smiles.cuda()
                b_seqs = b_seqs.float().cuda()
                b_affinity = b_affinity.float().cuda()
                output = self.model(b_smiles, b_seqs)
                if params['loss'] == 'MSE':
                    all_predictions.append(output.cpu().numpy())
                    all_labels.append(b_affinity.cpu().numpy())
                else:
                    all_predictions.append(torch.softmax(output, dim=1)[:, 1].cpu().numpy())
                    all_labels.append(b_affinity.cpu().numpy())
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        if params['loss'] == 'MSE':
            mse_value = self.mse(all_labels, all_predictions)
            ci_value = concordance_index(all_labels, all_predictions)
            print(f'validation mse: {mse_value}, validation ci: {ci_value}')
        else:
            auc = roc_auc_score(all_labels.astype(int), all_predictions)
            aupr = average_precision_score(all_labels.astype(int), all_predictions)
            precision = precision_score(all_labels.astype(int), (all_predictions > 0.5).astype(int))
            recall = recall_score(all_labels.astype(int), (all_predictions > 0.5).astype(int))
            print(f'auc: {auc}, aupr: {aupr}, precision: {precision}, recall: {recall}')
        return all_predictions


class MyDataset(Dataset):
    def __init__(self, smiles, seqs, seqs_mask, labels):
        assert len(smiles) == len(seqs) == len(labels), "SMILES, target, and binding affinity should correspond one-to-one!"
        self.smiles = smiles
        self.seqs_mask = seqs_mask
        self.seqs = seqs
        self.labels = torch.from_numpy(labels)

    def __getitem__(self, index):
        return self.smiles[index], self.seqs[index], self.seqs_mask[index], self.labels[index]

    def __len__(self):
        return len(self.smiles)


class Sample:
    def __init__(self, batch_size=128, fileName=None):
        data = torch.load(f'tmp/dataset_{fileName}.pt')
        self.dataloader = DataLoader(data, batch_size=batch_size, shuffle='training' in fileName)