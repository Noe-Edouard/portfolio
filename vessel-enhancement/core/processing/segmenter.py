import numpy as np
from numpy import ndarray
from typing import Literal, Optional, Tuple
from sklearn.metrics import precision_recall_curve

from core.utils.helpers import normalize_data

class Segmenter:
    
    def __init__(self):
        self.selector = {
            'thresholding': self.thresholding
        }
       
    def select_segmentation_function(self, method: Literal['thresholding']):
        if method not in self.selector:
            raise ValueError(f"Unknown segmentation method: {method}. Valid methods: {[key for key, value in self.selector.items()]}")

        return self.selector[method]
    
    def thresholding(
        self, 
        data: ndarray, 
        threshold: float = 0.5, 
        ground_truth: Optional[ndarray] = None
    ) -> Tuple[ndarray, float]:
        # https://sirineamrane.medium.com/from-auc-roc-to-optimal-treshold-selection-a-guide-for-binary-classification-679bae8ea1bf
        data_normalized = normalize_data(data)
        if ground_truth is not None:
            gt_normalized = normalize_data(ground_truth)
            precision, recall, thresholds = precision_recall_curve(gt_normalized.ravel(), data_normalized.ravel())
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            threshold = thresholds[np.argmax(f1_scores)]
        elif threshold is None:
            threshold = 0.5
        data_segmented = (data_normalized > threshold).astype(np.uint8)
        return data_segmented, threshold



    