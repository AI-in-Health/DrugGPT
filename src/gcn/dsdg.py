import logging
from sentence_transformers import SentenceTransformer
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

logging.basicConfig(level=logging.INFO)


class DSDGGenerator:
    def __init__(self, excel_path, embd_model_name='all-MiniLM-L6-v2', tau=0.1, k=5):
        """
            A class for generating and managing a Disease-Symptom-Drug Graph (DSDG).

            This class creates a graph representation of diseases and drugs, along with their associated properties,
            and provides methods for graph operations and embeddings calculation.

            Attributes:
                excel_path (str): Path to the Excel file containing disease and drug information.
                embd_model (SentenceTransformer): The embedding model used to encode text data.
                tau (float): Temperature parameter for softmax calculation in distance metric.
                k (int): The number of top-k elements to consider in certain calculations.
                G (networkx.Graph): The generated graph representing diseases and drugs.
                df (pandas.DataFrame): Store the dataframe to avoid reading twice
                initialized (bool): Flag to check if the graph is initialized

            Methods:
                generate_dsdg_dict(): Generates a dictionary from the Excel data for diseases and drugs.
                get_average_symptoms_embedding(): Calculates the average embedding of all symptoms in the dataset.
                get_knowledge_category(name, category): Retrieves specific knowledge category data for a given entity.
                calculate_distance(embedding1, embedding2): Computes a distance metric between two embeddings.
                get_top_k(arr): Returns indices of top-k elements in an array.
                _initialize_graph(): Initializes and populates the graph based on the dsdg_dict data.
                get_graph(): Returns the constructed graph.
        """

        self.excel_path = excel_path
        
        if not os.path.exists(excel_path):
            logging.error(f"Excel file not found at path: {excel_path}")
            raise FileNotFoundError(f"Excel file not found at path: {excel_path}")
            
        
        if tau <= 0:
            logging.warning(f"Invalid tau value: {tau}. Setting to default 0.1")
            tau = 0.1
            
        
        if not isinstance(k, int) or k <= 0:
            logging.warning(f"Invalid k value: {k}. Setting to default 5")
            k = 5
            
        self.tau = tau
        self.k = k
        self.G = nx.Graph()
        self.df = None
        self.dsdg_dict = {}
        self.initialized = False
        
        
        try:
            self.embd_model = SentenceTransformer(embd_model_name)
            logging.info(f"Initialized SentenceTransformer model: {embd_model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize SentenceTransformer: {e}")
            raise
        
        
        try:
            self.dsdg_dict = self.generate_dsdg_dict()
            self._initialize_graph()
            self.initialized = True
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise

    def generate_dsdg_dict(self):
        """
        Generates a dictionary from the Excel data for diseases and drugs.
        
        Returns:
            dict: Dictionary containing disease and drug information
        """
        logging.info("Generating DSDG dictionary from Excel.")
        try:
            
            try:
                self.df = pd.read_excel(self.excel_path)
            except Exception as e:
                logging.error(f"Failed to read Excel file {self.excel_path}: {e}")
                raise
                
            if self.df.empty:
                logging.error("Excel file is empty")
                raise ValueError("Excel file is empty. Cannot proceed with an empty dataset.")
            
            
            required_columns = ['Disease', 'Drug']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                error_msg = f"Required column(s) {missing_columns} not found in Excel file"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            
            dsdg_dict = {}
            for _, row in self.df.iterrows():
                disease = row['Disease']
                drug = row['Drug']
                
                
                if pd.isna(disease) or pd.isna(drug):
                    logging.warning(f"Skipping row with null disease or drug: {row}")
                    continue
                
                
                disease_categories = {}
                for cat in ['symptoms', 'causes', 'diagnosis', 'treatment', 'complications']:
                    cat_key = f"Disease {cat}"
                    if cat_key in row and not pd.isna(row[cat_key]):
                        disease_categories[cat_key] = row[cat_key]
                    else:
                        disease_categories[cat_key] = ""  
                dsdg_dict[disease] = disease_categories
                
                
                drug_categories = {}
                for cat in ['description', 'dosage', 'effects', 'toxicity', 'food_interaction', 'drug_interaction',
                           'pharmacodynamics', 'experimental_results']:
                    cat_key = f"Drug {cat}"
                    if cat_key in row and not pd.isna(row[cat_key]):
                        drug_categories[cat_key] = row[cat_key]
                    else:
                        drug_categories[cat_key] = ""  
                dsdg_dict[drug] = drug_categories
            
            if not dsdg_dict:
                logging.warning("Generated DSDG dictionary is empty")
                
            return dsdg_dict
            
        except Exception as e:
            logging.error(f"Error generating DSDG dictionary: {e}")
            raise

    def get_average_symptoms_embedding(self):
        """
        Calculates the average embedding of all symptoms in the dataset.
        
        Returns:
            numpy.ndarray: Average embedding vector
        """
        symptom_embeddings = []
        try:
            for disease, categories in self.dsdg_dict.items():
                if 'Disease symptoms' in categories and categories['Disease symptoms']:
                    symptoms = categories['Disease symptoms']
                    
                    symptom_embedding = self.embd_model.encode(symptoms)
                    symptom_embeddings.append(symptom_embedding)

            
            if symptom_embeddings:
                avg_embedding = np.mean(symptom_embeddings, axis=0)
            else:
                logging.warning("No symptom data found, returning zero embedding")
                avg_embedding = np.zeros(self.embd_model.get_sentence_embedding_dimension())

            return avg_embedding
            
        except Exception as e:
            logging.error(f"Error calculating average symptoms embedding: {e}")
            
            return np.zeros(self.embd_model.get_sentence_embedding_dimension())

    def get_knowledge_category(self, name, category):
        """
        Retrieves specific knowledge category data for a given entity.
        
        Args:
            name (str): Entity name (disease or drug)
            category (str): Category of knowledge to retrieve
            
        Returns:
            str: The requested knowledge or 'Unknown' if not found
        """
        try:
            
            if name in self.dsdg_dict:
                return self.dsdg_dict[name].get(category, 'Unknown')
            return 'Unknown'
        except Exception as e:
            logging.error(f"Error retrieving knowledge category: {e}")
            return 'Unknown'

    def calculate_distance(self, embedding1, embedding2):
        """
        Computes a distance metric between two embeddings.
        
        Args:
            embedding1 (numpy.ndarray): First embedding
            embedding2 (numpy.ndarray): Second embedding
            
        Returns:
            float or numpy.ndarray: Distance metric
        """
        try:
            
            if embedding1 is None or embedding2 is None:
                logging.error("Cannot calculate distance with None embeddings")
                raise ValueError("Embeddings cannot be None")
                
            
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)
                
            
            if embedding1.shape[1] != embedding2.shape[1]:
                logging.error(f"Embedding dimension mismatch: {embedding1.shape} vs {embedding2.shape}")
                raise ValueError(f"Embedding dimensions must match: {embedding1.shape} vs {embedding2.shape}")
                
            
            cosine_sim = cosine_similarity(embedding1, embedding2)
            
            
            
            tau = max(self.tau, 1e-6)
            
            
            
            max_val = np.max(cosine_sim)
            exp_values = np.exp((cosine_sim - max_val) / tau)
            denominator = np.sum(exp_values)
            
            
            if denominator < 1e-10:
                logging.warning("Near-zero denominator in softmax calculation")
                denominator = 1e-10
                
            
            result = exp_values / denominator
            
            
            if np.isnan(result).any():
                logging.warning("NaN values detected in distance calculation")
                result = np.nan_to_num(result, nan=0.0)
                
            return result
            
        except Exception as e:
            logging.error(f"Error calculating distance: {e}")
            
            if embedding1.ndim > 1:
                return np.zeros((embedding1.shape[0], embedding2.shape[0]))
            else:
                return 0.0

    def get_top_k(self, arr):
        """
        Returns indices of top-k elements in an array.
        
        Args:
            arr (numpy.ndarray): Input array
            
        Returns:
            numpy.ndarray: Indices of top-k elements
        """
        try:
            
            if arr is None or len(arr) == 0:
                logging.warning("Empty array provided to get_top_k")
                return np.array([])
                
            
            k = min(self.k, len(arr))
            
            
            if k <= 0:
                logging.warning(f"Invalid k value: {k}, returning empty array")
                return np.array([])
                
            
            if len(arr) <= k:
                return np.arange(len(arr))
                
            
            return arr.argsort()[-k:][::-1]
            
        except Exception as e:
            logging.error(f"Error getting top-k elements: {e}")
            return np.array([])

    def _initialize_graph(self):
        """
        Initializes and populates the graph based on the dsdg_dict data.
        This should only be called once during initialization.
        """
        if self.initialized:
            logging.warning("Graph already initialized. Skipping re-initialization.")
            return
            
        logging.info("Initializing graph from Excel.")
        try:
            
            if self.df is None:
                logging.warning("DataFrame not loaded. Loading Excel file again.")
                try:
                    self.df = pd.read_excel(self.excel_path)
                except Exception as e:
                    logging.error(f"Failed to read Excel file: {e}")
                    raise
                
            
            if self.df.empty:
                logging.error("DataFrame is empty, cannot initialize graph")
                raise ValueError("Cannot initialize graph from empty DataFrame")
                
            try:
                self.diseases = self.df['Disease'].unique()
                self.drugs = self.df['Drug'].unique()
            except KeyError as e:
                logging.error(f"Required column not found: {e}")
                raise

            logging.info("Creating disease and drug nodes with embeddings.")
            
            for disease in self.diseases:
                try:
                    
                    description = self.get_knowledge_category(disease, 'Disease description')
                    if description == 'Unknown':
                        
                        description = self.get_knowledge_category(disease, 'Disease symptoms')
                    
                    if description and description != 'Unknown':
                        embedding = self.embd_model.encode(description)
                        self.G.add_node(disease, embedding=embedding, type='disease')
                    else:
                        
                        embedding = np.zeros(self.embd_model.get_sentence_embedding_dimension())
                        self.G.add_node(disease, embedding=embedding, type='disease')
                        logging.warning(f"No description found for disease: {disease}")
                except Exception as e:
                    logging.error(f"Error creating node for disease {disease}: {e}")
                    
                    embedding = np.zeros(self.embd_model.get_sentence_embedding_dimension())
                    self.G.add_node(disease, embedding=embedding, type='disease')

            
            for drug in self.drugs:
                try:
                    
                    description = self.get_knowledge_category(drug, 'Drug description')
                    if description and description != 'Unknown':
                        embedding = self.embd_model.encode(description)
                        self.G.add_node(drug, embedding=embedding, type='drug')
                    else:
                        
                        embedding = np.zeros(self.embd_model.get_sentence_embedding_dimension())
                        self.G.add_node(drug, embedding=embedding, type='drug')
                        logging.warning(f"No description found for drug: {drug}")
                except Exception as e:
                    logging.error(f"Error creating node for drug {drug}: {e}")
                    
                    embedding = np.zeros(self.embd_model.get_sentence_embedding_dimension())
                    self.G.add_node(drug, embedding=embedding, type='drug')

            logging.info("Creating edges between disease and drug nodes.")
            for _, row in self.df.iterrows():
                try:
                    disease = row['Disease']
                    drug = row['Drug']
                    
                    
                    if pd.isna(disease) or pd.isna(drug):
                        continue
                    
                    
                    if not self.G.has_node(disease) or not self.G.has_node(drug):
                        continue
                        
                    disease_embedding = self.G.nodes[disease]['embedding']
                    drug_embedding = self.G.nodes[drug]['embedding']
                    distance = self.calculate_distance(disease_embedding, drug_embedding)
                    
                    if hasattr(distance, "shape") and len(distance.shape) > 0:
                        distance = distance[0, 0]
                    self.G.add_edge(disease, drug, weight=distance)
                except Exception as e:
                    logging.error(f"Error creating edge between {disease} and {drug}: {e}")

            logging.info("Updating edges based on top-K.")
            for disease in self.diseases:
                try:
                    if not self.G.has_node(disease):
                        continue
                        
                    neighbors = list(self.G.neighbors(disease))
                    if not neighbors:
                        continue
                        
                    neighbor_embeddings = np.array([self.G.nodes[neighbor]['embedding'] for neighbor in neighbors])
                    disease_embedding = self.G.nodes[disease]['embedding']
                    
                    
                    distances = self.calculate_distance(disease_embedding.reshape(1, -1), neighbor_embeddings)
                    
                    
                    if hasattr(distances, "shape") and len(distances.shape) > 1:
                        distances = distances.flatten()
                        
                    top_k_indices = self.get_top_k(distances)
                    
                    
                    if len(top_k_indices) == 0:
                        continue

                    for idx, neighbor in enumerate(neighbors):
                        if idx not in top_k_indices:
                            self.G.remove_edge(disease, neighbor)
                except Exception as e:
                    logging.error(f"Error updating edges for disease {disease}: {e}")
                    
        except Exception as e:
            logging.error(f"Error initializing graph: {e}")
            raise

    def get_graph(self):
        """
        Returns the constructed graph.
        
        Returns:
            networkx.Graph: The disease-drug graph
        """
        if not self.initialized:
            logging.warning("Graph not initialized yet")
            
            try:
                self._initialize_graph()
                self.initialized = True
            except Exception as e:
                logging.error(f"Failed to initialize graph on demand: {e}")
        return self.G
