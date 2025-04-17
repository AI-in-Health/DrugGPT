import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import logging


logger = logging.getLogger(__name__)


class GraphConvolutionalLayer(nn.Module):
    """
    Implementation of a Graph Convolutional Layer.
    
    This layer performs graph convolution operations as described in the paper
    "Semi-Supervised Classification with Graph Convolutional Networks" by Kipf and Welling.
    
    Attributes:
        weight (nn.Parameter): Learnable weight matrix for the layer
        bias (nn.Parameter, optional): Learnable bias vector
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionalLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Glorot initialization and biases with zeros."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Forward pass through the GCN layer.
        
        Args:
            x (torch.Tensor): Node feature matrix
            adj (torch.Tensor): Adjacency matrix of the graph
            
        Returns:
            torch.Tensor: Output feature matrix
        """
        support = torch.mm(x, self.weight)
        
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphConvolutionalNetwork(nn.Module):
    """
    Graph Convolutional Network (GCN) for generating soft prompts.
    
    This model combines multiple GCN layers to learn representations of nodes in a graph,
    which can be used to create soft prompts for language models.
    
    Attributes:
        gc1 (GraphConvolutionalLayer): First GCN layer
        gc2 (GraphConvolutionalLayer): Second GCN layer
        dropout (float): Dropout rate
        device (torch.device): Device to use for computation
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GraphConvolutionalNetwork, self).__init__()
        self.gc1 = GraphConvolutionalLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvolutionalLayer(hidden_dim, output_dim)
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)
        
    def forward(self, x, adj):
        """
        Forward pass through the GCN.
        
        Args:
            x (torch.Tensor): Node feature matrix
            adj (torch.Tensor): Adjacency matrix of the graph
            
        Returns:
            torch.Tensor: Output embeddings for all nodes
        """
        
        x = x.to(self.device)
        adj = adj.to(self.device)
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

    def generate_prefix(self, graph, identified_entities, max_length=50):
        """
        Generate a soft prompt prefix based on identified entities in the graph.
        
        Args:
            graph (networkx.Graph): The knowledge graph
            identified_entities (list): List of entity names identified in the query
            max_length (int): Maximum length of the generated prefix
            
        Returns:
            torch.Tensor: Generated soft prompt prefix
            
        Raises:
            ValueError: If graph is empty or no valid entities found
            RuntimeError: If there are embedding dimension mismatches
        """
        
        if not graph or len(graph.nodes()) == 0:
            logger.error("Empty graph provided to generate_prefix")
            raise ValueError("Empty graph provided to generate_prefix. A valid knowledge graph is required.")
                
        
        if not identified_entities:
            logger.error("No entities provided to generate_prefix")
            raise ValueError("No entities provided to generate_prefix. At least one entity is required.")
            
        
        adj = nx.to_numpy_array(graph)
        adj = torch.FloatTensor(adj)
        
        
        try:
            adj = self._normalize_adj(adj)
        except Exception as e:
            logger.error(f"Failed to normalize adjacency matrix: {str(e)}")
            raise RuntimeError(f"Failed to normalize adjacency matrix: {str(e)}")
        
        
        node_features = []
        expected_dim = self.gc1.in_features
        dimension_errors = 0
        
        for node in graph.nodes():
            if 'embedding' in graph.nodes[node]:
                embedding = graph.nodes[node]['embedding']
                
                if len(embedding) != expected_dim:
                    dimension_errors += 1
                    logger.warning(f"Node {node} has embedding dimension mismatch. Expected {expected_dim}, got {len(embedding)}.")
                    
                    if len(embedding) > expected_dim:
                        embedding = embedding[:expected_dim]
                    else:
                        
                        padding = np.zeros(expected_dim - len(embedding))
                        embedding = np.concatenate([embedding, padding])
                node_features.append(embedding)
            else:
                
                logger.warning(f"Node {node} has no embedding.")
                node_features.append(np.zeros(expected_dim))
        
        
        if dimension_errors > 0:
            logger.warning(f"Found {dimension_errors} nodes with incorrect embedding dimensions")
        
        
        if not node_features:
            logger.error("No valid node features found")
            raise ValueError("No valid node features found in the graph")
        
        
        x = torch.FloatTensor(np.array(node_features))
        
        
        embeddings = self.forward(x, adj)
        
        
        prefix = torch.zeros(1, max_length, self.gc2.out_features).to(self.device)
        
        
        entity_indices = []
        node_list = list(graph.nodes())
        
        
        found_entities = []
        for entity in identified_entities:
            if entity in graph:
                try:
                    entity_idx = node_list.index(entity)
                    entity_indices.append(entity_idx)
                    found_entities.append(entity)
                except ValueError:
                    logger.error(f"Entity {entity} not found in graph nodes list.")
            else:
                logger.warning(f"Entity '{entity}' not found in the graph")
        
        
        if not entity_indices:
            missing_entities = set(identified_entities) - set(found_entities)
            error_msg = f"None of the identified entities {identified_entities} were found in the graph."
            if missing_entities:
                error_msg += f" Missing entities: {list(missing_entities)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        
        entity_embeddings = embeddings[entity_indices]
        averaged_embedding = torch.mean(entity_embeddings, dim=0).unsqueeze(0)
        
        
        if torch.isnan(averaged_embedding).any():
            logger.error("NaN values detected in averaged embedding")
            
            raise RuntimeError("NaN values detected in averaged embedding. Check GCN training or normalize input features.")
        
        
        prefix_length = min(max_length, len(entity_indices) * 5)  
        for i in range(prefix_length):
            prefix[0, i] = averaged_embedding
            
        logger.info(f"Generated prefix with length {prefix_length} for entities: {found_entities}")
        return prefix
        
    def _normalize_adj(self, adj):
        """
        Normalize adjacency matrix using symmetric normalization.
        
        Args:
            adj (torch.Tensor): Adjacency matrix
            
        Returns:
            torch.Tensor: Normalized adjacency matrix
            
        Raises:
            RuntimeError: If normalization fails due to invalid adjacency matrix
        """
        
        adj = adj + torch.eye(adj.size(0))
        
        
        d = torch.sum(adj, dim=1)
        
        
        if torch.min(d) == 0:
            isolated_nodes = (d == 0).nonzero(as_tuple=True)[0]
            logger.warning(f"Found {len(isolated_nodes)} isolated nodes in the graph")
            
            d = d + 1e-10
            
        d_inv_sqrt = torch.pow(d, -0.5)
        
        
        if torch.isinf(d_inv_sqrt).any() or torch.isnan(d_inv_sqrt).any():
            logger.error("Numerical issues in adjacency matrix normalization: infinite or NaN values")
            problematic_indices = torch.where(torch.isinf(d_inv_sqrt) | torch.isnan(d_inv_sqrt))[0]
            logger.error(f"Problematic indices: {problematic_indices}")
            
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
            d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
            
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        
        try:
            normalized_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        except RuntimeError as e:
            logger.error(f"Matrix multiplication error during normalization: {str(e)}")
            raise RuntimeError(f"Failed to normalize adjacency matrix: {str(e)}")
        
        
        if torch.isnan(normalized_adj).any():
            nan_count = torch.isnan(normalized_adj).sum().item()
            logger.error(f"Found {nan_count} NaN values in normalized adjacency matrix")
            
            raise RuntimeError(f"Normalized adjacency matrix contains {nan_count} NaN values")
        
        return normalized_adj
