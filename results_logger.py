import csv
import os

class ResultsLogger:
    def __init__(self, filepath='results_database.csv'):
        self.filepath = filepath
        self.fieldnames = [
            'Category', 
            'Technique_Used', 
            'Strength', 
            'Guidance', 
            'L1', 
            'L2',
            'MS_SSIM', 
            'LPIPS', 
            'Max_Patch', 
            'Label' # 0 for Good, 1 for Anomaly
        ]
        self._initialize_file()

    def _initialize_file(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_result(self, result_dict):
        """
        Logs a single test run result.
        Ensure result_dict contains keys matching self.fieldnames.
        """
        with open(self.filepath, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            # Filter out any extra keys from result_dict not in fieldnames
            filtered_row = {k: v for k, v in result_dict.items() if k in self.fieldnames}
            writer.writerow(filtered_row)
