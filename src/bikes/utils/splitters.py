"""Split the dataframes into subsets"""

# imports
import abc
import typing as T

import numpy as np
import numpy.typing as npt
import pydantic as pdt
from sklearn import model_selection

from bikes.core import schemas

# TYPES
Index = npt.NDArray[np.int64]
TrainTestIndex = tuple[Index, Index]
TrainTestSplits = T.Iterator[TrainTestIndex]

# splitters


class Splitter(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a splitter.

    Use splitters to split data in sets
    """

    KIND: str

    @abc.abstractmethod
    def split(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> TrainTestSplits:
        """Split a dataframe into subsets.




        Args:
            inputs (schemas.Inputs) : model inputs.
            targets (schemas.Targets): model targets.
            groups (Index | None, optional) : group labels.




        Returns:
            TrainTestSplit: iterator over the dataframe train/test splits.
        """

    @abc.abstractmethod
    def get_n_splits(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> int:
        """Get the number of splits generated.


        Args:
            inputs (schemas.Inputs): models inputs.
            targets (schemas.Targets): model targets.
            groups (Index | None, optional): group labels.


        Returns:
            int: number of splits generated

        """


class TrainTestSplitter(Splitter):
    """Split a dataframe into a train and test set.

    Parameters:
        shuffle (bool): shuffle the dataset. Default is False.
        test_size (int | float): number/ratio for the test set.
        random_state (int): random state for the splitter object.



    """

    KIND: T.Literal["TrainTestSplitter"] = "TrainTestSplitter"

    shuffle: bool = False  # since it is time sensitive
    test_size: int | float = 24 * 30 * 2  # for two months
    random_state: int = 42

    @T.override
    def split(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> TrainTestSplits:
        index = np.arange(len(inputs))
