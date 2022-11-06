# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from dataclasses import dataclass, field
from typing import List, Optional, Type, Iterator

import numpy as np

from statsforecast.models import (
    ADIDA,
    AutoARIMA,
    AutoCES,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    ETS,
    IMAPA,
    TSB,
)

from gluonts.core.component import validated
from gluonts.dataset import DataEntry, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import forecast_start
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.model.predictor import RepresentablePredictor
from gluonts.model.forecast import QuantileForecast
from gluonts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
)


@dataclass
class ModelConfig:
    quantile_levels: Optional[List[float]] = None
    forecast_keys: List[str] = field(init=False)
    statsforecast_keys: List[str] = field(init=False)
    intervals: Optional[List[int]] = field(init=False)

    def __post_init__(self):
        self.forecast_keys = ["mean"]
        self.statsforecast_keys = ["mean"]
        if self.quantile_levels is None:
            self.intervals = None
            return

        intervals = set()

        for quantile_level in self.quantile_levels:
            interval = round(
                200 * (max(quantile_level, 1 - quantile_level) - 0.5)
            )
            intervals.add(interval)
            side = "hi" if quantile_level > 0.5 else "lo"
            self.forecast_keys.append(str(quantile_level))
            self.statsforecast_keys.append(f"{side}-{interval}")

        self.intervals = sorted(intervals)


class StatsForecastPredictor(RepresentablePredictor):
    """
    A predictor type that wraps models from the `statsforecast`_ package.

    This class is used via subclassing and setting the ``ModelType`` class
    attribute to specify the ``statsforecast`` model type to use.

    .. _statsforecast: https://github.com/Nixtla/statsforecast

    Parameters
    ----------
    prediction_length
        Prediction length for the model to use.
    quantile_levels
        Optional list of quantile levels that we want predictions for.
        Note: this is only supported by specific types of models, such as
        ``AutoARIMA``. By default this is ``None``, giving only the mean
        prediction.
    **model_params
        Keyword arguments to be passed to the model type for construction.
        The specific arguments accepted or required depend on the
        ``ModelType``; please refer to the documentation of ``statsforecast``
        for details.
    """

    ModelType: Type

    @validated()
    def __init__(
        self,
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
        num_feat_dynamic_real: int = 0,
        num_feat_static_real: int = 0,
        freq: Optional[str] = None,
        time_features: Optional[List[TimeFeature]] = None,
        **model_params,
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        self.model = self.ModelType(**model_params)
        self.config = ModelConfig(quantile_levels=quantile_levels)
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_real = num_feat_static_real
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(freq)
        )

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                    )
                ]
                if not self.num_feat_static_real > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.num_feat_dynamic_real > 0
                        else []
                    ),
                ),
            ]
        )

    def predict(
        self, dataset: Dataset, **kwargs
    ) -> Iterator[QuantileForecast]:
        transformation = self.create_transformation()
        transformed_dataset = transformation(dataset, is_train=False)
        for item in transformed_dataset:
            yield self.predict_item(item)

    def predict_item(self, entry: DataEntry) -> QuantileForecast:
        kwargs = {}
        if self.config.intervals is not None:
            kwargs["level"] = self.config.intervals

        time_feat = entry[FieldName.FEAT_TIME].T
        static_real_cov = entry[FieldName.FEAT_STATIC_REAL]
        repeat_static_real_cov = static_real_cov[None, :].repeat(
            time_feat.shape[0], axis=0
        )
        covariates = np.hstack([time_feat, repeat_static_real_cov])

        prediction = self.model.forecast(
            y=entry[FieldName.TARGET],
            X=covariates[: -self.prediction_length, :],
            X_future=covariates[-self.prediction_length :, :],
            h=self.prediction_length,
            **kwargs,
        )

        forecast_arrays = [
            prediction[k] for k in self.config.statsforecast_keys
        ]

        return QuantileForecast(
            forecast_arrays=np.stack(forecast_arrays, axis=0),
            forecast_keys=self.config.forecast_keys,
            start_date=forecast_start(entry),
            item_id=entry.get("item_id"),
            info=entry.get("info"),
        )


class ADIDAPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``ADIDA`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = ADIDA


class AutoARIMAPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``AutoARIMA`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = AutoARIMA


class AutoCESPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``AutoCES`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = AutoCES


class CrostonClassicPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``CrostonClassic`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = CrostonClassic


class CrostonOptimizedPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``CrostonOptimized`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = CrostonOptimized


class CrostonSBAPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``CrostonSBA`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = CrostonSBA


class ETSPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``ETS`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = ETS


class IMAPAPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``IMAPA`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = IMAPA


class TSBPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``TSB`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = TSB