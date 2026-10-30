
"""Provide task-specific estimators, predictions and evaluation metrics."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
import pandas as pd

# Import ml_utils lazily to avoid circular import
# ml_utils imports get_task_handler from this module


class TaskHandler(ABC):
    """Base class for task-specific handlers."""

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Return the task name: 'classification' or 'regression'."""
        pass

    @abstractmethod
    def prepare_y(self, y: pd.Series) -> np.ndarray:
        """Prepare target values for training/prediction."""
        pass

    @abstractmethod
    def predict(self, pipe, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using the pipeline.

        Returns:
            dict with keys like 'y_pred', 'y_prob' (for classification), etc.
        """
        pass

    @abstractmethod
    def compute_metrics(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray], threshold: float = 0.5) -> Dict[str, Any]:
        """Compute task-specific metrics."""
        pass

    @abstractmethod
    def create_pred_df(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        id_col: str,
        elements_col: str,
        group_col: str,
        y_col: str,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Create prediction DataFrame with task-specific columns."""
        pass

    @property
    def requires_threshold(self) -> bool:
        """Whether this task uses a threshold parameter."""
        return False


class BinaryClassificationHandler(TaskHandler):
    """Handler for binary classification tasks."""

    @property
    def task_name(self) -> str:
        return "classification"

    def prepare_y(self, y: pd.Series) -> np.ndarray:
        return y.astype(int).values

    def predict(self, pipe, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        y_prob = pipe.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        return {
            "y_prob": y_prob,
            "y_pred": y_pred,
        }

    def compute_metrics(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray], threshold: float = 0.5) -> Dict[str, Any]:
        # Lazy import to avoid circular dependency
        from . import ml_utils
        y_prob = predictions["y_prob"]
        return ml_utils.compute_metrics(y_true, y_prob, threshold=threshold)

    def create_pred_df(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        id_col: str,
        elements_col: str,
        group_col: str,
        y_col: str,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        y_prob = predictions["y_prob"]
        y_pred = (y_prob >= threshold).astype(int)

        return pd.DataFrame({
            id_col: df[id_col] if id_col in df.columns else None,
            "filepath": df["filepath"] if "filepath" in df.columns else None,
            y_col: y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
            elements_col: df[elements_col] if elements_col in df.columns else None,
            group_col: df[group_col] if group_col in df.columns else None,
        })

    @property
    def requires_threshold(self) -> bool:
        return True


class RegressionHandler(TaskHandler):
    """Handler for regression tasks."""

    @property
    def task_name(self) -> str:
        return "regression"

    def prepare_y(self, y: pd.Series) -> np.ndarray:
        return y.astype(float).values

    def predict(self, pipe, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        y_pred = pipe.predict(X)
        return {
            "y_pred": y_pred.astype(float),
        }

    def compute_metrics(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray], threshold: float = 0.5) -> Dict[str, Any]:
        # Lazy import to avoid circular dependency
        from . import ml_utils
        y_pred = predictions["y_pred"]
        return ml_utils.compute_regression_metrics(y_true, y_pred)

    def create_pred_df(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        id_col: str,
        elements_col: str,
        group_col: str,
        y_col: str,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        y_pred = predictions["y_pred"]

        return pd.DataFrame({
            id_col: df[id_col] if id_col in df.columns else None,
            "filepath": df["filepath"] if "filepath" in df.columns else None,
            y_col: y_true,
            "y_pred": y_pred,
            elements_col: df[elements_col] if elements_col in df.columns else None,
            group_col: df[group_col] if group_col in df.columns else None,
        })

    @property
    def requires_threshold(self) -> bool:
        return False


def get_task_handler(task: str) -> TaskHandler:
    """
    Factory function to get the appropriate task handler.

    Parameters
    ----------
    task : str
        Task type: "classification" or "regression"

    Returns
    -------
    TaskHandler
        Appropriate handler instance
    """
    task = str(task).lower().strip()

    if task == "classification":
        return BinaryClassificationHandler()
    elif task == "regression":
        return RegressionHandler()
    else:
        raise ValueError(f"Unsupported task: {task}. Use 'classification' or 'regression'.")
