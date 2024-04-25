from typing import List
import numpy as np

class Input:
    def __init__(self, task_type: int, matrix: List[List[float]], vector: List[float]) -> None:
        self.task_type = task_type
        self.matrix = np.array(matrix)
        self.vector = np.array(vector)