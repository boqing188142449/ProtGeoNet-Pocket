import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, BatchNorm


class TNet(nn.Module):
    """Transformation Network: Learns geometric transformations for point clouds."""
    def __init__(self, input_dim=3):
        super(TNet, self).__init__()
        self.input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim * input_dim)
        )
        self.reg_loss = 0.0

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]
        x = self.fc(x).view(batch_size, self.input_dim, self.input_dim)

        # Initialize as identity matrix
        init_transform = torch.eye(self.input_dim, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        transform = init_transform + x

        # Orthogonal regularization
        self.reg_loss = torch.mean(
            torch.norm(torch.bmm(transform, transform.transpose(1, 2)) - init_transform, dim=(1, 2)))
        return transform


class PointNet(nn.Module):
    """PointNet: Processes point clouds with multi-scale feature extraction and attention."""
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=1024):
        super(PointNet, self).__init__()
        self.input_dim = input_dim
        self.tnet = TNet(input_dim=input_dim)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim * 2, 1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.attn = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim * 3, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        transform = self.tnet(x)
        x = torch.bmm(x.transpose(1, 2), transform).transpose(1, 2)

        x1 = self.mlp1(x)
        x2 = self.mlp2(x)
        x3 = self.mlp3(x)

        multi_feat = torch.cat([x1, x2, x3], dim=1)
        attn_weights = self.attn(multi_feat)
        multi_feat = multi_feat * attn_weights
        out = self.proj(multi_feat)
        return out


class GNN(nn.Module):
    """Graph Neural Network: Processes graph-structured features with residual connections."""
    def __init__(self, in_features, hidden_features=1024, dropout=0.3, edge_dim=20):
        super(GNN, self).__init__()

        def make_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )

        self.conv1 = GINEConv(make_mlp(in_features, hidden_features), edge_dim=edge_dim)
        self.bn1 = BatchNorm(hidden_features)
        self.res1 = nn.Identity() if in_features == hidden_features else nn.Linear(in_features, hidden_features)

        self.conv2 = GINEConv(make_mlp(hidden_features, hidden_features // 2), edge_dim=edge_dim)
        self.bn2 = BatchNorm(hidden_features // 2)
        self.res2 = nn.Linear(hidden_features, hidden_features // 2)

        self.conv3 = GINEConv(make_mlp(hidden_features // 2, hidden_features // 8), edge_dim=edge_dim)
        self.bn3 = BatchNorm(hidden_features // 8)
        self.res3 = nn.Linear(hidden_features // 2, hidden_features // 8)

        self.conv4 = GINEConv(make_mlp(hidden_features // 8, hidden_features // 16), edge_dim=edge_dim)
        self.bn4 = BatchNorm(hidden_features // 16)
        self.res4 = nn.Linear(hidden_features // 8, hidden_features // 16)

        self.fc_g1 = nn.Linear(hidden_features // 16, hidden_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_features)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.size(1) == 0:  # Handle empty graphs
            return self.ln(x)

        x1 = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)) + self.res1(x))
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index, edge_attr)) + self.res2(x1))
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index, edge_attr)) + self.res3(x2))
        x4 = F.relu(self.bn4(self.conv4(x3, edge_index, edge_attr)) + self.res4(x3))

        out = self.relu(self.fc_g1(x4))
        out = self.dropout(out)
        out = self.ln(out)
        return out


class CombinedModel(nn.Module):
    """Combined Model: Integrates PointNet and GNN for per-amino-acid predictions."""
    def __init__(self, input_feature, hidden_features=1024, out_features=1, dropout=0.3):
        super(CombinedModel, self).__init__()
        self.pointnet = PointNet(input_dim=3, hidden_dim=hidden_features, output_dim=hidden_features)
        self.gnn = GNN(in_features=hidden_features, hidden_features=hidden_features, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_features + input_feature, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, out_features)
        )
        self.sigmoid = nn.Sigmoid() if out_features == 1 else nn.Identity()

    def forward(self, data):
        features = data.x
        coords = data.protein_pos.unsqueeze(0)

        point_feat = self.pointnet(coords)
        point_feat = point_feat.squeeze(0).permute(1, 0)

        combined_feat = torch.cat([point_feat, features], dim=-1)
        combined_feat = self.fusion(combined_feat)

        edge_attr = data.edge_attr
        if edge_attr is None:
            raise ValueError("data.edge_attr is None, expected shape (num_edges, 20)")

        gnn_feat = self.gnn(combined_feat, data.edge_index, edge_attr)
        out = self.fc_out(gnn_feat)
        return self.sigmoid(out)