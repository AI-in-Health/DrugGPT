import pandas as pd
from tqdm import tqdm
import logging


class Evaluation:
    def __init__(self, ensemble_model, parser_dict, log_results=True, store_results=False,
                 log_wrong_answers_only=False, useopenai=False):
        self.ensemble_model = ensemble_model
        self.log_results = log_results
        self.parser_dict = parser_dict
        self.useopenai = useopenai
        self.store_results = store_results
        self.log_wrong_answers_only = log_wrong_answers_only

    @staticmethod
    def check_text_accuracy(prediction, actual):
        prediction = prediction.rstrip('.')
        prediction_initials = ''.join(word[0] for word in prediction.split())
        actual_words = actual.split()
        return all(word in prediction for word in actual_words) or prediction_initials == actual

    @staticmethod
    def calculate_f1_metrics(prediction, label):
        prediction_set = set(prediction.split(', '))
        label_set = set(label.split(', '))
        true_positives = len(prediction_set & label_set)
        false_positives = len(prediction_set - label_set)
        false_negatives = len(label_set - prediction_set)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return precision, recall, f1_score

    def log_answer(self, i, input_data, prediction, actual_label):
        if self.log_wrong_answers_only and prediction.lower() == actual_label.lower():
            return
        logging.info(f"Index: {i} Question: {input_data}")
        logging.info(f"Predicted: {prediction}, Actual: {actual_label}")

    def evaluate(self, dataset_name, evaluation_set):
        print(f"\nEvaluating: {dataset_name}")
        slice_size = len(evaluation_set['sample'])

        accurate_predictions = 0
        precision_list = []
        recall_list = []
        f1_list = []
        wrong_answers = []

        for i in tqdm(range(slice_size), desc="Processing"):
            input_data = evaluation_set['sample'][i]
            full_response = self.ensemble_model.run_inference(input_data, use_openai=self.useopenai)
            parsed_response = self.parser_dict[dataset_name](full_response)

            correct_answer = self.check_text_accuracy(parsed_response, evaluation_set['label'][i].lower())
            if correct_answer:
                accurate_predictions += 1
            else:
                wrong_answers.append((i, input_data, parsed_response, evaluation_set['label'][i]))

            if dataset_name == 'chatDoctor':
                precision, recall, f1 = self.calculate_f1_metrics(parsed_response, evaluation_set['label'][i].lower())
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

            self.log_answer(i, input_data, parsed_response, evaluation_set['label'][i])

        results = {
            'Accuracy': accurate_predictions / slice_size
        }

        if dataset_name == 'chatDoctor':
            results['Average Precision'] = sum(precision_list) / len(precision_list)
            results['Average Recall'] = sum(recall_list) / len(recall_list)
            results['Average F1 Score'] = sum(f1_list) / len(f1_list)

        if self.store_results:
            df = pd.DataFrame(wrong_answers, columns=['Index', 'Question', 'Predicted', 'Actual'])
            df.to_csv(f'evaluation_wrong_answers_{dataset_name}.csv', index=False)

        return results
