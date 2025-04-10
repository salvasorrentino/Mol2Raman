import torch
import torch.nn.functional as F 

from torch.nn import Linear, BatchNorm1d, ModuleList, Sequential, ReLU
from torch_geometric.nn import TransformerConv, TopKPooling, GINEConv 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool

torch.manual_seed(42)
#

class GNN(torch.nn.Module):
    def __init__(self, 
                 feature_size, 
                 model_params,
                 n_data_points=1600):
        
        super(GNN, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(feature_size, 
                                     embedding_size,
                                     heads=n_heads,
                                     dropout=dropout_rate,
                                     edge_dim=edge_dim,
                                     beta=True)

        self.transf1 = Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        # Other layers
        for i in range(self.n_layers):
            
            self.conv_layers.append(TransformerConv(embedding_size, 
                                                    embedding_size, 
                                                    heads=n_heads, 
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))

            self.transf_layers.append(Linear(embedding_size*n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))

            if i % self.top_k_every_n == 0:

                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        # Linear layers
        self.linear1 = Linear(embedding_size*2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons/2))  
        self.linear3 = Linear(int(dense_neurons/2), n_data_points)  

    def forward(self, x, edge_attr, edge_index, batch_index):
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x , edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                    )
                # Add current representation
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
    
        x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.linear3(x)

        return torch.nn.Softplus()(x)
    

class GINE(torch.nn.Module):

    def __init__(self, 
                 node_feature_size, 
                 edge_feature_size,
                 dim_h, 
                 n_data_points,
                 additional_feature_size=12):
        super(GINE, self).__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(node_feature_size, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
                       edge_dim=edge_feature_size)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
                       edge_dim=edge_feature_size
                       )
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
                       edge_dim=edge_feature_size)
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
                       edge_dim=edge_feature_size)


        self.lin1 = Linear(dim_h*4, dim_h*4)
        self.lin2 = Linear(dim_h*4, n_data_points)

        self.global_proj = Linear(1, dim_h + 1)

    def forward(self, x, graph_level_feats, edge_attr, edge_index, batch_index):
        # Node embeddings

        h1 = self.conv1(x, edge_index, edge_attr)
        # h1.shape = [batch_size, dim_h=64]
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        h4 = self.conv4(h2, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch_index)
        h2 = global_add_pool(h2, batch_index)
        h3 = global_add_pool(h3, batch_index)
        h4 = global_add_pool(h4, batch_index)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3, h4), dim=1)
        # h = torch.cat((h1, h2), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.25, training=self.training)
        h = self.lin2(h)
        
        return torch.nn.Softplus()(h)


class GINEGLOBAL(torch.nn.Module):

    def __init__(self,
                 node_feature_size,
                 edge_feature_size,
                 dim_h,
                 n_data_points,
                 additional_feature_size=12):
        super(GINEGLOBAL, self).__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(node_feature_size, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size
        )
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)

        self.lin1 = Linear(dim_h * 4 + additional_feature_size, dim_h * 32)
        self.lin3 = Linear(dim_h * 32, dim_h * 16)
        self.lin2 = Linear(dim_h * 16, n_data_points)

        # self.global_proj = Linear(1, dim_h + 1)

    def forward(self, x, graph_level_feats, edge_attr, edge_index, batch_index):
        # Node embeddings

        h1 = self.conv1(x, edge_index, edge_attr)
        # h1.shape = [batch_size, dim_h=64]
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        h4 = self.conv4(h3, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch_index)
        h2 = global_add_pool(h2, batch_index)
        h3 = global_add_pool(h3, batch_index)
        h4 = global_add_pool(h4, batch_index)

        # Concatenate graph embeddings
        skip_features = graph_level_feats.reshape(h1.shape[0], -1).type(torch.float32)
        h = torch.cat((h1, h2, h3, h4, skip_features), dim=1)
        # h = torch.cat((h1, h2, h3, skip_features), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.4, training=self.training)
        h = self.lin3(h)
        h = h.relu()
        h = F.dropout(h, p=0.4, training=self.training)
        h = self.lin2(h)

        return torch.nn.Softplus()(h)


class ClassModel(torch.nn.Module):

    def __init__(self,
                 node_feature_size,
                 edge_feature_size,
                 dim_h,
                 n_data_points,
                 ):
        super(ClassModel, self).__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(node_feature_size, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size
        )
        # self.conv3 = GINEConv(
        #     Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
        #                Linear(dim_h, dim_h), ReLU()),
        #     edge_dim=edge_feature_size)
        # self.conv4 = GINEConv(
        #     Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
        #                Linear(dim_h, dim_h), ReLU()),
        #     edge_dim=edge_feature_size)
        self.lin1 = Linear(dim_h * 2, dim_h * 64)
        self.lin2 = Linear(dim_h * 64, n_data_points)

    def forward(self, x, graph_level_feats, edge_attr, edge_index, batch_index):
        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        # h3 = self.conv3(h2, edge_index, edge_attr)
        # h4 = self.conv4(h2, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch_index)
        h2 = global_add_pool(h2, batch_index)
        #h3 = global_add_pool(h3, batch_index)
        # h4 = global_add_pool(h4, batch_index)

        # Concatenate graph embeddings
        # h = torch.cat((h1, h2, h3, h4), dim=1)
        h = torch.cat((h1, h2), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.25, training=self.training)
        h = self.lin2(h)

        return torch.nn.Sigmoid()(h)


class ClassModelGlobal(torch.nn.Module):

    def __init__(self,
                 node_feature_size,
                 edge_feature_size,
                 dim_h,
                 n_data_points,
                 ):
        super(ClassModelGlobal, self).__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(node_feature_size, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size
        )
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.lin1 = Linear(dim_h * 4 + 1, dim_h * 64)
        self.lin2 = Linear(dim_h * 64, n_data_points)

    def forward(self, x, graph_level_feats, edge_attr, edge_index, batch_index):
        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        h4 = self.conv4(h2, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch_index)
        h2 = global_add_pool(h2, batch_index)
        h3 = global_add_pool(h3, batch_index)
        h4 = global_add_pool(h4, batch_index)

        # Concatenate graph embeddings
        skip_features = graph_level_feats.reshape(h1.shape[0], -1).type(torch.float32)
        h = torch.cat((h1, h2, h3, h4, skip_features), dim=1)


        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.25, training=self.training)
        h = self.lin2(h)

        return torch.nn.Sigmoid()(h)


class ModelPredPos(torch.nn.Module):

    def __init__(self,
                 node_feature_size,
                 edge_feature_size,
                 dim_h,
                 n_data_points,
                 additional_feature_size=12):
        super(ModelPredPos, self).__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(node_feature_size, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size
        )
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.lin1 = Linear(dim_h * 4, dim_h * 4)
        self.lin2 = Linear(dim_h * 4, n_data_points)

    def forward(self, x, graph_level_feats, edge_attr, edge_index, batch_index):
        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        # h1.shape = [batch_size, dim_h=64]
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        h4 = self.conv4(h2, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch_index)
        h2 = global_add_pool(h2, batch_index)
        h3 = global_add_pool(h3, batch_index)
        h4 = global_add_pool(h4, batch_index)

        # Concatenate graph embeddings
        # skip_features = graph_level_feats.reshape(h1.shape[0], -1).type(torch.float32)
        h = torch.cat((h1, h2, h3, h4), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.25, training=self.training)
        h = self.lin2(h)

        # output_pos = self.lin2(h)
        # output_al = self.lin2(h)

        return torch.nn.Softplus()(h)
        # return [output_pos, output_al]


class ModelPredNumPeak(torch.nn.Module):

    def __init__(self,
                 node_feature_size,
                 edge_feature_size,
                 dim_h,
                 n_data_points,
                 additional_feature_size=12):
        super(ModelPredNumPeak, self).__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(node_feature_size, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size
        )
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.lin1 = Linear(dim_h * 4, dim_h * 4)
        self.lin2 = Linear(dim_h * 4, n_data_points)

    def forward(self, x, graph_level_feats, edge_attr, edge_index, batch_index):
        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        # h1.shape = [batch_size, dim_h=64]
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        h4 = self.conv4(h3, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch_index)
        h2 = global_add_pool(h2, batch_index)
        h3 = global_add_pool(h3, batch_index)
        h4 = global_add_pool(h4, batch_index)

        # Concatenate graph embeddings
        skip_features = graph_level_feats.reshape(h1.shape[0], -1).type(torch.float32)
        # h = torch.cat((h1, h2, h3, h4, skip_features), dim=1)
        h = torch.cat((h1, h2, h3, h4), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.25, training=self.training)
        h = self.lin2(h)

        # output_pos = self.lin2(h)
        # output_al = self.lin2(h)

        return torch.nn.Softplus()(h)
        # return [output_pos, output_al]
