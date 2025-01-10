"""Trainnable machine learning models"""

# imports

import abc
import typing as T
from typing import Self

import pydantic as pdt
import shap
from sklearn import compose, ensemble, pipeline, preprocessing

from bikes.core import schemas

# Model params
ParamKey = str
ParamValue = T.Any
Params = dict[ParamKey, ParamValue]


class Model(abc.ABC, pdt.BaseModel, strict=True, frozen=False, extra="forbid"):
    """Base class for a project model"""

    KIND: str

    def get_params(self, deep: bool = True) -> Params:
        """Get the model params.

        Args:
            deep (bool, optional): ignored.

        Returns:
            Params: internal model parameters.



        """
        params: Params = {}

        for key, value in self.model_dump().items():
            if not key.startswith("_") and not key.isupper():
                params[key] = value
        return params

    def set_params(self, **params: ParamValue) -> T.Self:
        """Set the model params in place

        Returns:
        T.Self: instance of the model:

        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @abc.abstractmethod
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets) -> Self:
        """Fit the model on the given inputs and targets
        Args:
            inputs (schemas.Inputs): model prediction inputs

        Returns:
            schemas.FeatureImportances: feature importances

        """

    def explain_model(self) -> schemas.FeatureImportances:
        """Explain the internal model structure

        Returns:
            schemas.FeatureImportances: feature importances.

        """
        raise NotImplementedError()

    def get_internal_model(self) -> T.Any:
        """Return the internal model in the object.


        Raises:
            NotImplementedError: method not implemented

        Returns:
            T.Any : any internal model(either empty or fitted)

        """
        raise NotImplementedError()


class BaselineSklearnModel(Model):
    """baseline model in scikit-learn

    Parameters:
        max_depth (int): maximum depth of the the random forest
        n_estimators (int): number of estimators in the random forest
        random_state (int,optional): random state of the machnine learning pipeline

    """

    KIND: T.Literal["BaselinesklearnModel"] = "BaselineSklearnModel"

    # parameters
    max_depth: int = 20
    n_estimators: int = 200
    random_state: int | None = 42

    # private
    _pipeline: pipeline.Pipeline | None = None
    _numericals: list[str] = [
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "casual",
        "registered",
    ]

    _categoricals: list[str] = [
        "season",
        "weathersit",
    ]

    @T.override
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets) -> "BaselineSklearnModel":
        # subcomponents
        categorical_transformer = preprocessing.OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )

        # components
        transformer = compose.ColumnTransformer(
            [
                ("categoricals", categorical_transformer, self._categoricals)(
                    "numericals", "passthrough", self._numericals
                )
            ],
            reminder="drop",
        )
        regressor = ensemble.RandomForestRegressor(
            max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=self.random_state
        )

        # pipeline
        self._pipeline = pipeline.Pipeline(
            steps=[
                ("transformer", transformer),
                ("regressor", regressor),
            ]
        )

        self._pipeline.fit(X=inputs, y=targets[schemas.TargetsSchema.cnt])
        return self

    @T.override
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        model = self.get_internal_model()
        prediction = model.predict(inputs)
        outputs = schemas.Outputs({schemas.OutputSchema.prediction: prediction})
