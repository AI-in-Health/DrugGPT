import logging
from sentence_transformers import SentenceTransformer
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
        self.embd_model = SentenceTransformer(embd_model_name)
        self.tau = tau
        self.k = k
        self.G = nx.Graph()
        self.dsdg_dict = self.generate_dsdg_dict()
        self._initialize_graph()

    def generate_dsdg_dict(self):
        logging.info("Generating DSDG dictionary from Excel.")
        df = pd.read_excel(self.excel_path)
        dsdg_dict = {}
        for _, row in df.iterrows():
            disease = row['Disease']
            drug = row['Drug']
            dsdg_dict[disease] = {f"Disease {cat}": row[f"Disease {cat}"] for cat in
                                  ['symptoms', 'causes', 'diagnosis', 'treatment', 'complications']}
            dsdg_dict[drug] = {f"Drug {cat}": row[f"Drug {cat}"] for cat in
                               ['description', 'dosage', 'effects', 'toxicity', 'food_interaction', 'drug_interaction',
                                'pharmacodynamics', 'experimental_results']}
        return dsdg_dict

    def get_average_symptoms_embedding(self):
        symptom_embeddings = []
        for disease, categories in self.dsdg_dict.items():
            if 'symptoms' in categories:
                symptoms = categories['symptoms']
                # Encode symptoms to embeddings
                symptom_embedding = self.embd_model.encode(symptoms)
                symptom_embeddings.append(symptom_embedding)

        # Calculate the average embedding
        if symptom_embeddings:
            avg_embedding = np.mean(symptom_embeddings, axis=0)
        else:
            avg_embedding = np.zeros(self.embd_model.get_sentence_embedding_dimension())

        return avg_embedding

    def get_knowledge_category(self, name, category):
        return self.dsdg_dict.get(name, {}).get(category, 'Unknown')

    def calculate_distance(self, embedding1, embedding2):
        cosine_sim = cosine_similarity(embedding1, embedding2)
        numerator = np.exp(cosine_sim / self.tau)
        denominator = np.sum(np.exp(cosine_sim / self.tau))
        return numerator / denominator

    def get_top_k(self, arr):
        return arr.argsort()[-self.k:][::-1]

    def _initialize_graph(self):
        logging.info("Initializing graph from Excel.")
        df = pd.read_excel(self.excel_path)
        self.diseases = df['Disease'].unique()
        self.drugs = df['Drug'].unique()

        logging.info("Creating disease and drug nodes with embeddings.")
        for disease in self.diseases:
            description = self.get_knowledge_category(disease, 'Disease description')
            embedding = self.embd_model.encode(description)
            self.G.add_node(disease, embedding=embedding, type='disease')

        for drug in self.drugs:
            description = self.get_knowledge_category(drug, 'Drug description')
            embedding = self.embd_model.encode(description)
            self.G.add_node(drug, embedding=embedding, type='drug')

        logging.info("Creating edges between disease and drug nodes.")
        for _, row in df.iterrows():
            disease = row['Disease']
            drug = row['Drug']
            disease_embedding = self.G.nodes[disease]['embedding']
            drug_embedding = self.G.nodes[drug]['embedding']
            distance = self.calculate_distance(disease_embedding, drug_embedding)
            self.G.add_edge(disease, drug, weight=distance)

        logging.info("Updating edges based on top-K.")
        for disease in self.diseases:
            neighbors = list(self.G.neighbors(disease))
            neighbor_embeddings = np.array([self.G.nodes[neighbor]['embedding'] for neighbor in neighbors])
            disease_embedding = self.G.nodes[disease]['embedding']
            distances = self.calculate_distance(disease_embedding, neighbor_embeddings)
            top_k_indices = self.get_top_k(distances)

            for idx, neighbor in enumerate(neighbors):
                if idx not in top_k_indices:
                    self.G.remove_edge(disease, neighbor)

    def get_graph(self):
        return self.G
