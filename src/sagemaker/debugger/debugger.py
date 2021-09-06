# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Amazon SageMaker Debugger provides full visibility into ML training jobs.

This module provides SageMaker Debugger high-level methods
to set up Debugger objects, such as Debugger built-in rules, tensor collections,
and hook configuration. Use the Debugger objects for parameters when constructing
a SageMaker estimator to initiate a training job.

"""
from __future__ import absolute_import

import time

from abc import ABC

import attr

import re
from datetime import datetime
from enum import Enum

import smdebug_rulesconfig as rule_configs

from sagemaker import image_uris
from sagemaker.utils import build_dict

from sagemaker.debugger.metrics_config import (
    DetailedProfilingConfig,
    DataloaderProfilingConfig,
    SMDataParallelProfilingConfig,
    HorovodProfilingConfig,
    PythonProfilingConfig,
)
from sagemaker.debugger.profiler_constants import (
    BASE_FOLDER_DEFAULT,
    CLOSE_FILE_INTERVAL_DEFAULT,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
    MAX_FILE_SIZE_DEFAULT,
)
from sagemaker.debugger.utils import ErrorMessages

ALL_METRIC_CONFIGS = [
    DetailedProfilingConfig,
    DataloaderProfilingConfig,
    PythonProfilingConfig,
    HorovodProfilingConfig,
    SMDataParallelProfilingConfig,
]

framework_name = "debugger"
DEBUGGER_FLAG = "USE_SMDEBUG"


def get_rule_container_image_uri(region):
    """Return the Debugger rule image URI for the given AWS Region.

    For a full list of rule image URIs,
    see `Use Debugger Docker Images for Built-in or Custom Rules
    <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-docker-images-rules.html>`_.

    Args:
        region (str): A string of AWS Region. For example, ``'us-east-1'``.

    Returns:
        str: Formatted image URI for the given AWS Region and the rule container type.

    """
    return image_uris.retrieve(framework_name, region)


def get_default_profiler_rule():
    """Return the default built-in profiler rule with a unique name.

    Returns:
        sagemaker.debugger.ProfilerRule: The instance of the built-in ProfilerRule.

    """
    default_rule = rule_configs.ProfilerReport()
    custom_name = f"{default_rule.rule_name}-{int(time.time())}"
    return ProfilerRule.sagemaker(default_rule, name=custom_name)


@attr.s
class RuleBase(ABC):
    """The SageMaker Debugger rule base class that cannot be instantiated directly.

    .. tip::

        Debugger rule classes inheriting this RuleBase class are
        :class:`~sagemaker.debugger.Rule` and :class:`~sagemaker.debugger.ProfilerRule`.
        Do not directly use the rule base class to instantiate a SageMaker Debugger rule.
        Use the :class:`~sagemaker.debugger.Rule` classmethods for debugging
        and the :class:`~sagemaker.debugger.ProfilerRule` classmethods for profiling.

    Attributes:
        name (str): The name of the rule.
        image_uri (str): The image URI to use the rule.
        instance_type (str): Type of EC2 instance to use. For example, 'ml.c4.xlarge'.
        container_local_output_path (str): The local path to store the Rule output.
        s3_output_path (str): The location in S3 to store the output.
        volume_size_in_gb (int): Size in GB of the EBS volume to use for storing data.
        rule_parameters (dict): A dictionary of parameters for the rule.

    """

    name = attr.ib()
    image_uri = attr.ib()
    instance_type = attr.ib()
    container_local_output_path = attr.ib()
    s3_output_path = attr.ib()
    volume_size_in_gb = attr.ib()
    rule_parameters = attr.ib()

    @staticmethod
    def _set_rule_parameters(source, rule_to_invoke, rule_parameters):
        """Create a dictionary of rule parameters.

        Args:
            source (str): Optional. A source file containing a rule to invoke. If provided,
                you must also provide rule_to_invoke. This can either be an S3 uri or
                a local path.
            rule_to_invoke (str): Optional. The name of the rule to invoke within the source.
                If provided, you must also provide source.
            rule_parameters (dict): Optional. A dictionary of parameters for the rule.

        Returns:
            dict: A dictionary of rule parameters.

        """
        if bool(source) ^ bool(rule_to_invoke):
            raise ValueError(
                "If you provide a source, you must also provide a rule to invoke (and vice versa)."
            )

        merged_rule_params = {}
        merged_rule_params.update(build_dict("source_s3_uri", source))
        merged_rule_params.update(build_dict("rule_to_invoke", rule_to_invoke))
        merged_rule_params.update(rule_parameters or {})

        return merged_rule_params


class Rule(RuleBase):
    """The SageMaker Debugger Rule class configures *debugging* rules to debug your training job.

    The debugging rules analyze tensor outputs from your training job
    and monitor conditions that are critical for the success of the training
    job.

    SageMaker Debugger comes pre-packaged with built-in *debugging* rules.
    For example, the debugging rules can detect whether gradients are getting too large or
    too small, or if a model is overfitting.
    For a full list of built-in rules for debugging, see
    `List of Debugger Built-in Rules
    <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.
    You can also write your own rules using the custom rule classmethod.

    """

    def __init__(
        self,
        name,
        image_uri,
        instance_type,
        container_local_output_path,
        s3_output_path,
        volume_size_in_gb,
        rule_parameters,
        collections_to_save,
        actions=None,
    ):
        """Configure the debugging rules using the following classmethods.

        .. tip::
            Use the following ``Rule.sagemaker`` class method for built-in debugging rules
            or the ``Rule.custom`` class method for custom debugging rules.
            Do not directly use the :class:`~sagemaker.debugger.Rule`
            initialization method.

        """
        super(Rule, self).__init__(
            name,
            image_uri,
            instance_type,
            container_local_output_path,
            s3_output_path,
            volume_size_in_gb,
            rule_parameters,
        )
        self.collection_configs = collections_to_save
        self.actions = actions

    @classmethod
    def sagemaker(
        cls,
        base_config,
        name=None,
        container_local_output_path=None,
        s3_output_path=None,
        other_trials_s3_input_paths=None,
        rule_parameters=None,
        collections_to_save=None,
        actions=None,
    ):
        """Initialize a ``Rule`` object for a *built-in* debugging rule.

        Args:
            base_config (dict): Required. This is the base rule config dictionary returned from the
                :class:`~sagemaker.debugger.rule_configs` method.
                For example, ``rule_configs.dead_relu()``.
                For a full list of built-in rules for debugging, see
                `List of Debugger Built-in Rules
                <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.
            name (str): Optional. The name of the debugger rule. If one is not provided,
                the name of the base_config will be used.
            container_local_output_path (str): Optional. The local path in the rule processing
                container.
            s3_output_path (str): Optional. The location in Amazon S3 to store the output tensors.
                The default Debugger output path for debugging data is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/debug-output/.
            other_trials_s3_input_paths ([str]): Optional. The Amazon S3 input paths
                of other trials to use the SimilarAcrossRuns rule.
            rule_parameters (dict): Optional. A dictionary of parameters for the rule.
            collections_to_save (:class:`~sagemaker.debugger.CollectionConfig`):
                Optional. A list
                of :class:`~sagemaker.debugger.CollectionConfig` objects to be saved.

        Returns:
            :class:`~sagemaker.debugger.Rule`: An instance of the built-in rule.

        **Example of how to create a built-in rule instance:**

        .. code-block:: python

            from sagemaker.debugger import Rule, rule_configs

            built_in_rules = [
                Rule.sagemaker(rule_configs.built_in_rule_name_in_pysdk_format_1()),
                Rule.sagemaker(rule_configs.built_in_rule_name_in_pysdk_format_2()),
                ...
                Rule.sagemaker(rule_configs.built_in_rule_name_in_pysdk_format_n())
            ]

        You need to replace the ``built_in_rule_name_in_pysdk_format_*`` with the
        names of built-in rules. You can find the rule names at `List of Debugger Built-in
        Rules <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.

        **Example of creating a built-in rule instance with adjusting parameter values:**

        .. code-block:: python

            from sagemaker.debugger import Rule, rule_configs

            built_in_rules = [
                Rule.sagemaker(
                    base_config=rule_configs.built_in_rule_name_in_pysdk_format(),
                    rule_parameters={
                            "key": "value"
                    }
                    collections_to_save=[
                        CollectionConfig(
                            name="tensor_collection_name",
                            parameters={
                                "key": "value"
                            }
                        )
                    ]
                )
            ]

        For more information about setting up the ``rule_parameters`` parameter,
        see `List of Debugger Built-in
        Rules <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.

        For more information about setting up the ``collections_to_save`` parameter,
        see the :class:`~sagemaker.debugger.CollectionConfig` class.

        """
        merged_rule_params = {}

        if rule_parameters is not None and rule_parameters.get("rule_to_invoke") is not None:
            raise RuntimeError(
                """You cannot provide a 'rule_to_invoke' for SageMaker rules.
                Either remove the rule_to_invoke or use a custom rule.

                """
            )

        if actions is not None and not rule_configs.is_valid_action_object(actions):
            raise RuntimeError("""`actions` must be of type `Action` or `ActionList`!""")

        if other_trials_s3_input_paths is not None:
            for index, s3_input_path in enumerate(other_trials_s3_input_paths):
                merged_rule_params["other_trial_{}".format(str(index))] = s3_input_path

        default_rule_params = base_config["DebugRuleConfiguration"].get("RuleParameters", {})
        merged_rule_params.update(default_rule_params)
        merged_rule_params.update(rule_parameters or {})

        base_config_collections = []
        for config in base_config.get("CollectionConfigurations", []):
            collection_name = None
            collection_parameters = {}
            for key, value in config.items():
                if key == "CollectionName":
                    collection_name = value
                if key == "CollectionParameters":
                    collection_parameters = value
            base_config_collections.append(
                CollectionConfig(name=collection_name, parameters=collection_parameters)
            )

        return cls(
            name=name or base_config["DebugRuleConfiguration"].get("RuleConfigurationName"),
            image_uri="DEFAULT_RULE_EVALUATOR_IMAGE",
            instance_type=None,
            container_local_output_path=container_local_output_path,
            s3_output_path=s3_output_path,
            volume_size_in_gb=None,
            rule_parameters=merged_rule_params,
            collections_to_save=collections_to_save or base_config_collections,
            actions=actions,
        )

    @classmethod
    def custom(
        cls,
        name,
        image_uri,
        instance_type,
        volume_size_in_gb,
        source=None,
        rule_to_invoke=None,
        container_local_output_path=None,
        s3_output_path=None,
        other_trials_s3_input_paths=None,
        rule_parameters=None,
        collections_to_save=None,
        actions=None,
    ):
        """Initialize a ``Rule`` object for a *custom* debugging rule.

        You can create a custom rule that analyzes tensors emitted
        during the training of a model
        and monitors conditions that are critical for the success of a training
        job. For more information, see `Create Debugger Custom Rules for Training Job
        Analysis
        <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-custom-rules.html>`_.

        Args:
            name (str): Required. The name of the debugger rule.
            image_uri (str): Required. The URI of the image to be used by the debugger rule.
            instance_type (str): Required. Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            volume_size_in_gb (int): Required. Size in GB of the EBS volume
                to use for storing data.
            source (str): Optional. A source file containing a rule to invoke. If provided,
                you must also provide rule_to_invoke. This can either be an S3 uri or
                a local path.
            rule_to_invoke (str): Optional. The name of the rule to invoke within the source.
                If provided, you must also provide source.
            container_local_output_path (str): Optional. The local path in the container.
            s3_output_path (str): Optional. The location in Amazon S3 to store the output tensors.
                The default Debugger output path for debugging data is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/debug-output/.
            other_trials_s3_input_paths ([str]): Optional. The Amazon S3 input paths
                of other trials to use the SimilarAcrossRuns rule.
            rule_parameters (dict): Optional. A dictionary of parameters for the rule.
            collections_to_save ([sagemaker.debugger.CollectionConfig]): Optional. A list
                of :class:`~sagemaker.debugger.CollectionConfig` objects to be saved.

        Returns:
            :class:`~sagemaker.debugger.Rule`: The instance of the custom rule.

        """
        if actions is not None and not rule_configs.is_valid_action_object(actions):
            raise RuntimeError("""`actions` must be of type `Action` or `ActionList`!""")

        merged_rule_params = cls._set_rule_parameters(
            source, rule_to_invoke, other_trials_s3_input_paths, rule_parameters
        )

        return cls(
            name=name,
            image_uri=image_uri,
            instance_type=instance_type,
            container_local_output_path=container_local_output_path,
            s3_output_path=s3_output_path,
            volume_size_in_gb=volume_size_in_gb,
            rule_parameters=merged_rule_params,
            collections_to_save=collections_to_save or [],
            actions=actions,
        )

    def prepare_actions(self, training_job_name):
        """Prepare actions for Debugger Rule.

        Args:
            training_job_name (str): The training job name. To be set as the default training job
                prefix for the StopTraining action if it is specified.
        """
        if self.actions is None:
            # user cannot manually specify action_json in rule_parameters for actions.
            self.rule_parameters.pop("action_json", None)
            return

        self.actions.update_training_job_prefix_if_not_specified(training_job_name)
        action_params = {"action_json": self.actions.serialize()}
        self.rule_parameters.update(action_params)

    @staticmethod
    def _set_rule_parameters(source, rule_to_invoke, other_trials_s3_input_paths, rule_parameters):
        """Set rule parameters for Debugger Rule.

        Args:
            source (str): Optional. A source file containing a rule to invoke. If provided,
                you must also provide rule_to_invoke. This can either be an S3 uri or
                a local path.
            rule_to_invoke (str): Optional. The name of the rule to invoke within the source.
                If provided, you must also provide source.
            other_trials_s3_input_paths ([str]): Optional. S3 input paths for other trials.
            rule_parameters (dict): Optional. A dictionary of parameters for the rule.

        Returns:
            dict: A dictionary of rule parameters.

        """
        merged_rule_params = {}
        if other_trials_s3_input_paths is not None:
            for index, s3_input_path in enumerate(other_trials_s3_input_paths):
                merged_rule_params["other_trial_{}".format(str(index))] = s3_input_path

        merged_rule_params.update(
            super(Rule, Rule)._set_rule_parameters(source, rule_to_invoke, rule_parameters)
        )
        return merged_rule_params

    def to_debugger_rule_config_dict(self):
        """Generates a request dictionary using the parameters provided when initializing object.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        debugger_rule_config_request = {
            "RuleConfigurationName": self.name,
            "RuleEvaluatorImage": self.image_uri,
        }

        debugger_rule_config_request.update(build_dict("InstanceType", self.instance_type))
        debugger_rule_config_request.update(build_dict("VolumeSizeInGB", self.volume_size_in_gb))
        debugger_rule_config_request.update(
            build_dict("LocalPath", self.container_local_output_path)
        )
        debugger_rule_config_request.update(build_dict("S3OutputPath", self.s3_output_path))
        debugger_rule_config_request.update(build_dict("RuleParameters", self.rule_parameters))

        return debugger_rule_config_request


class ProfilerRule(RuleBase):
    """The SageMaker Debugger ProfilerRule class configures *profiling* rules.

    SageMaker Debugger profiling rules automatically analyze
    hardware system resource utilization and framework metrics of a
    training job to identify performance bottlenecks.

    SageMaker Debugger comes pre-packaged with built-in *profiling* rules.
    For example, the profiling rules can detect if GPUs are underutilized due to CPU bottlenecks or
    IO bottlenecks.
    For a full list of built-in rules for debugging, see
    `List of Debugger Built-in Rules <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.
    You can also write your own profiling rules using the Amazon SageMaker
    Debugger APIs.

    .. tip::
        Use the following ``ProfilerRule.sagemaker`` class method for built-in profiling rules
        or the ``ProfilerRule.custom`` class method for custom profiling rules.
        Do not directly use the `Rule` initialization method.

    """

    @classmethod
    def sagemaker(
        cls,
        base_config,
        name=None,
        container_local_output_path=None,
        s3_output_path=None,
    ):
        """Initialize a ``ProfilerRule`` object for a *built-in* profiling rule.

        The rule analyzes system and framework metrics of a given
        training job to identify performance bottlenecks.

        Args:
            base_config (rule_configs.ProfilerRule): The base rule configuration object
                returned from the ``rule_configs`` method.
                For example, 'rule_configs.ProfilerReport()'.
                For a full list of built-in rules for debugging, see
                `List of Debugger Built-in Rules
                <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.

            name (str): The name of the profiler rule. If one is not provided,
                the name of the base_config will be used.
            container_local_output_path (str): The path in the container.
            s3_output_path (str): The location in Amazon S3 to store the profiling output data.
                The default Debugger output path for profiling data is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/profiler-output/.

        Returns:
            :class:`~sagemaker.debugger.ProfilerRule`:
            The instance of the built-in ProfilerRule.

        """
        return cls(
            name=name or base_config.rule_name,
            image_uri="DEFAULT_RULE_EVALUATOR_IMAGE",
            instance_type=None,
            container_local_output_path=container_local_output_path,
            s3_output_path=s3_output_path,
            volume_size_in_gb=None,
            rule_parameters=base_config.rule_parameters,
        )

    @classmethod
    def custom(
        cls,
        name,
        image_uri,
        instance_type,
        volume_size_in_gb,
        source=None,
        rule_to_invoke=None,
        container_local_output_path=None,
        s3_output_path=None,
        rule_parameters=None,
    ):
        """Initialize a ``ProfilerRule`` object for a *custom* profiling rule.

        You can create a rule that
        analyzes system and framework metrics emitted during the training of a model and
        monitors conditions that are critical for the success of a
        training job.

        Args:
            name (str): The name of the profiler rule.
            image_uri (str): The URI of the image to be used by the proflier rule.
            instance_type (str): Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data.
            source (str): A source file containing a rule to invoke. If provided,
                you must also provide rule_to_invoke. This can either be an S3 uri or
                a local path.
            rule_to_invoke (str): The name of the rule to invoke within the source.
                If provided, you must also provide the source.
            container_local_output_path (str): The path in the container.
            s3_output_path (str): The location in Amazon S3 to store the output.
                The default Debugger output path for profiling data is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/profiler-output/.
            rule_parameters (dict): A dictionary of parameters for the rule.

        Returns:
            :class:`~sagemaker.debugger.ProfilerRule`:
            The instance of the custom ProfilerRule.

        """
        merged_rule_params = super()._set_rule_parameters(source, rule_to_invoke, rule_parameters)

        return cls(
            name=name,
            image_uri=image_uri,
            instance_type=instance_type,
            container_local_output_path=container_local_output_path,
            s3_output_path=s3_output_path,
            volume_size_in_gb=volume_size_in_gb,
            rule_parameters=merged_rule_params,
        )

    def to_profiler_rule_config_dict(self):
        """Generates a request dictionary using the parameters provided when initializing object.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        profiler_rule_config_request = {
            "RuleConfigurationName": self.name,
            "RuleEvaluatorImage": self.image_uri,
        }

        profiler_rule_config_request.update(build_dict("InstanceType", self.instance_type))
        profiler_rule_config_request.update(build_dict("VolumeSizeInGB", self.volume_size_in_gb))
        profiler_rule_config_request.update(
            build_dict("LocalPath", self.container_local_output_path)
        )
        profiler_rule_config_request.update(build_dict("S3OutputPath", self.s3_output_path))

        if self.rule_parameters:
            profiler_rule_config_request["RuleParameters"] = self.rule_parameters
            for k, v in profiler_rule_config_request["RuleParameters"].items():
                profiler_rule_config_request["RuleParameters"][k] = str(v)

        return profiler_rule_config_request


class DebuggerHookConfig(object):
    """Create a Debugger hook configuration object to save the tensor for debugging.

    DebuggerHookConfig provides options to customize how debugging
    information is emitted and saved. This high-level DebuggerHookConfig class
    runs based on the `smdebug.SaveConfig
    <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/
    api.md#saveconfig>`_ class.

    """

    def __init__(
        self,
        s3_output_path=None,
        container_local_output_path=None,
        hook_parameters=None,
        collection_configs=None,
    ):
        """Initialize the DebuggerHookConfig instance.

        Args:
            s3_output_path (str): Optional. The location in Amazon S3 to store the output tensors.
                The default Debugger output path is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/debug-output/.
            container_local_output_path (str): Optional. The local path in the container.
            hook_parameters (dict): Optional. A dictionary of parameters.
            collection_configs ([sagemaker.debugger.CollectionConfig]): Required. A list
                of :class:`~sagemaker.debugger.CollectionConfig` objects to be saved
                at the **s3_output_path**.

        **Example of creating a DebuggerHookConfig object:**

        .. code-block:: python

            from sagemaker.debugger import CollectionConfig, DebuggerHookConfig

            collection_configs=[
                CollectionConfig(name="tensor_collection_1")
                CollectionConfig(name="tensor_collection_2")
                ...
                CollectionConfig(name="tensor_collection_n")
            ]

            hook_config = DebuggerHookConfig(
                collection_configs=collection_configs
            )

        """
        self.s3_output_path = s3_output_path
        self.container_local_output_path = container_local_output_path
        self.hook_parameters = hook_parameters
        self.collection_configs = collection_configs

    def _to_request_dict(self):
        """Generate a request dictionary using the parameters when initializing the object.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        debugger_hook_config_request = {"S3OutputPath": self.s3_output_path}

        if self.container_local_output_path is not None:
            debugger_hook_config_request["LocalPath"] = self.container_local_output_path

        if self.hook_parameters is not None:
            debugger_hook_config_request["HookParameters"] = self.hook_parameters

        if self.collection_configs is not None:
            debugger_hook_config_request["CollectionConfigurations"] = [
                collection_config._to_request_dict()
                for collection_config in self.collection_configs
            ]

        return debugger_hook_config_request


class TensorBoardOutputConfig(object):
    """Create a tensor ouput configuration object for debugging visualizations on TensorBoard."""

    def __init__(self, s3_output_path, container_local_output_path=None):
        """Initialize the TensorBoardOutputConfig instance.

        Args:
            s3_output_path (str): Optional. The location in Amazon S3 to store the output.
            container_local_output_path (str): Optional. The local path in the container.

        """
        self.s3_output_path = s3_output_path
        self.container_local_output_path = container_local_output_path

    def _to_request_dict(self):
        """Generate a request dictionary using the instances attributes.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        tensorboard_output_config_request = {"S3OutputPath": self.s3_output_path}

        if self.container_local_output_path is not None:
            tensorboard_output_config_request["LocalPath"] = self.container_local_output_path

        return tensorboard_output_config_request


class CollectionConfig(object):
    """Creates tensor collections for SageMaker Debugger."""

    def __init__(self, name, parameters=None):
        """Constructor for collection configuration.

        Args:
            name (str): Required. The name of the collection configuration.
            parameters (dict): Optional. The parameters for the collection
                configuration.

        **Example of creating a CollectionConfig object:**

        .. code-block:: python

            from sagemaker.debugger import CollectionConfig

            collection_configs=[
                CollectionConfig(name="tensor_collection_1")
                CollectionConfig(name="tensor_collection_2")
                ...
                CollectionConfig(name="tensor_collection_n")
            ]

        For a full list of Debugger built-in collection, see
        `Debugger Built in Collections
        <https://github.com/awslabs/sagemaker-debugger/blob/master
        /docs/api.md#built-in-collections>`_.

        **Example of creating a CollectionConfig object with parameter adjustment:**

        You can use the following CollectionConfig template in two ways:
        (1) to adjust the parameters of the built-in tensor collections,
        and (2) to create custom tensor collections.

        If you put the built-in collection names to the ``name`` parameter,
        ``CollectionConfig`` takes it to match the built-in collections and adjust parameters.
        If you specify a new name to the ``name`` parameter,
        ``CollectionConfig`` creates a new tensor collection, and you must use
        ``include_regex`` parameter to specify regex of tensors you want to collect.

        .. code-block:: python

            from sagemaker.debugger import CollectionConfig

            collection_configs=[
                CollectionConfig(
                    name="tensor_collection",
                    parameters={
                        "key_1": "value_1",
                        "key_2": "value_2"
                        ...
                        "key_n": "value_n"
                    }
                )
            ]

        The following list shows the available CollectionConfig parameters.

        +--------------------------+---------------------------------------------------------+
        | Parameter Key            | Descriptions                                            |
        +==========================+=========================================================+
        |``include_regex``         |  Specify a list of regex patterns of tensors to save.   |
        |                          |                                                         |
        |                          |  Tensors whose names match these patterns will be saved.|
        +--------------------------+---------------------------------------------------------+
        |``save_histogram``        |  Set *True* if want to save histogram output data for   |
        |                          |                                                         |
        |                          |  TensorFlow visualization.                              |
        +--------------------------+---------------------------------------------------------+
        |``reductions``            |  Specify certain reduction values of tensors.           |
        |                          |                                                         |
        |                          |  This helps reduce the amount of data saved and         |
        |                          |                                                         |
        |                          |  increase training speed.                               |
        |                          |                                                         |
        |                          |  Available values are ``min``, ``max``, ``median``,     |
        |                          |                                                         |
        |                          |  ``mean``, ``std``, ``variance``, ``sum``, and ``prod``.|
        +--------------------------+---------------------------------------------------------+
        |``save_interval``         |  Specify how often to save tensors in steps.            |
        |                          |                                                         |
        |``train.save_interval``   |  You can also specify the save intervals                |
        |                          |                                                         |
        |``eval.save_interval``    |  in TRAIN, EVAL, PREDICT, and GLOBAL modes.             |
        |                          |                                                         |
        |``predict.save_interval`` |  The default value is 500 steps.                        |
        |                          |                                                         |
        |``global.save_interval``  |                                                         |
        +--------------------------+---------------------------------------------------------+
        |``save_steps``            |  Specify the exact step numbers to save tensors.        |
        |                          |                                                         |
        |``train.save_steps``      |  You can also specify the save steps                    |
        |                          |                                                         |
        |``eval.save_steps``       |  in TRAIN, EVAL, PREDICT, and GLOBAL modes.             |
        |                          |                                                         |
        |``predict.save_steps``    |                                                         |
        |                          |                                                         |
        |``global.save_steps``     |                                                         |
        +--------------------------+---------------------------------------------------------+
        |``start_step``            |  Specify the exact start step to save tensors.          |
        |                          |                                                         |
        |``train.start_step``      |  You can also specify the start steps                   |
        |                          |                                                         |
        |``eval.start_step``       |  in TRAIN, EVAL, PREDICT, and GLOBAL modes.             |
        |                          |                                                         |
        |``predict.start_step``    |                                                         |
        |                          |                                                         |
        |``global.start_step``     |                                                         |
        +--------------------------+---------------------------------------------------------+
        |``end_step``              |  Specify the exact end step to save tensors.            |
        |                          |                                                         |
        |``train.end_step``        |  You can also specify the end steps                     |
        |                          |                                                         |
        |``eval.end_step``         |  in TRAIN, EVAL, PREDICT, and GLOBAL modes.             |
        |                          |                                                         |
        |``predict.end_step``      |                                                         |
        |                          |                                                         |
        |``global.end_step``       |                                                         |
        +--------------------------+---------------------------------------------------------+

        For example, the following code shows how to control the save_interval parameters
        of the built-in ``losses`` tensor collection. With the following collection configuration,
        Debugger collects loss values every 100 steps from training loops and every 10 steps
        from evaluation loops.

        .. code-block:: python

            collection_configs=[
                CollectionConfig(
                    name="losses",
                    parameters={
                        "train.save_interval": "100",
                        "eval.save_interval": "10"
                    }
                )
            ]

        """
        self.name = name
        self.parameters = parameters

    def __eq__(self, other):
        """Equal method override.

        Args:
            other: Object to test equality against.

        """
        if not isinstance(other, CollectionConfig):
            raise TypeError(
                "CollectionConfig is only comparable with other CollectionConfig objects."
            )

        return self.name == other.name and self.parameters == other.parameters

    def __ne__(self, other):
        """Not-equal method override.

        Args:
            other: Object to test equality against.

        """
        if not isinstance(other, CollectionConfig):
            raise TypeError(
                "CollectionConfig is only comparable with other CollectionConfig objects."
            )

        return self.name != other.name or self.parameters != other.parameters

    def __hash__(self):
        """Hash method override."""
        return hash((self.name, tuple(sorted((self.parameters or {}).items()))))

    def _to_request_dict(self):
        """Generate a request dictionary using the parameters initializing the object.

        Returns:
            dict: A portion of an API request as a dictionary.

        """
        collection_config_request = {"CollectionName": self.name}

        if self.parameters is not None:
            collection_config_request["CollectionParameters"] = self.parameters

        return collection_config_request


class FrameworkProfile:
    """Sets up the profiling configuration for framework metrics.

    Validates user inputs and fills in default values if no input is provided.
    There are three main profiling options to choose from:
    :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig`,
    :class:`~sagemaker.debugger.metrics_config.DataloaderProfilingConfig`, and
    :class:`~sagemaker.debugger.metrics_config.PythonProfilingConfig`.

    The following list shows available scenarios of configuring the profiling options.

    1. None of the profiling configuration, step range, or time range is specified.
    SageMaker Debugger activates framework profiling based on the default settings
    of each profiling option.

    .. code-block:: python

        from sagemaker.debugger import ProfilerConfig, FrameworkProfile

        profiler_config=ProfilerConfig(
            framework_profile_params=FrameworkProfile()
        )

    2. Target step or time range is specified to
    this :class:`~sagemaker.debugger.metrics_config.FrameworkProfile` class.
    The requested target step or time range setting propagates to all of
    the framework profiling options.
    For example, if you configure this class as following, all of the profiling options
    profiles the 6th step:

    .. code-block:: python

        from sagemaker.debugger import ProfilerConfig, FrameworkProfile

        profiler_config=ProfilerConfig(
            framework_profile_params=FrameworkProfile(start_step=6, num_steps=1)
        )

    3. Individual profiling configurations are specified through
    the ``*_profiling_config`` parameters.
    SageMaker Debugger profiles framework metrics only for the specified profiling configurations.
    For example, if the :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig` class
    is configured but not the other profiling options, Debugger only profiles based on the settings
    specified to the
    :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig` class.
    For example, the following example shows a profiling configuration to perform
    detailed profiling at step 10, data loader profiling at step 9 and 10,
    and Python profiling at step 12.

    .. code-block:: python

        from sagemaker.debugger import ProfilerConfig, FrameworkProfile

        profiler_config=ProfilerConfig(
            framework_profile_params=FrameworkProfile(
                detailed_profiling_config=DetailedProfilingConfig(start_step=10, num_steps=1),
                dataloader_profiling_config=DataloaderProfilingConfig(start_step=9, num_steps=2),
                python_profiling_config=PythonProfilingConfig(start_step=12, num_steps=1),
            )
        )

    If the individual profiling configurations are specified in addition to
    the step or time range,
    SageMaker Debugger prioritizes the individual profiling configurations and ignores
    the step or time range. For example, in the following code,
    the ``start_step=1`` and ``num_steps=10`` will be ignored.

    .. code-block:: python

        from sagemaker.debugger import ProfilerConfig, FrameworkProfile

        profiler_config=ProfilerConfig(
            framework_profile_params=FrameworkProfile(
                start_step=1,
                num_steps=10,
                detailed_profiling_config=DetailedProfilingConfig(start_step=10, num_steps=1),
                dataloader_profiling_config=DataloaderProfilingConfig(start_step=9, num_steps=2),
                python_profiling_config=PythonProfilingConfig(start_step=12, num_steps=1)
            )
        )

    """

    def __init__(
        self,
        local_path=BASE_FOLDER_DEFAULT,
        file_max_size=MAX_FILE_SIZE_DEFAULT,
        file_close_interval=CLOSE_FILE_INTERVAL_DEFAULT,
        file_open_fail_threshold=FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
        detailed_profiling_config=None,
        dataloader_profiling_config=None,
        python_profiling_config=None,
        horovod_profiling_config=None,
        smdataparallel_profiling_config=None,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
    ):
        """Initialize the FrameworkProfile class object.

        Args:
            detailed_profiling_config (DetailedProfilingConfig): The configuration for detailed
                profiling. Configure it using the
                :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig` class.
                Pass ``DetailedProfilingConfig()`` to use the default configuration.
            dataloader_profiling_config (DataloaderProfilingConfig): The configuration for
                dataloader metrics profiling. Configure it using the
                :class:`~sagemaker.debugger.metrics_config.DataloaderProfilingConfig` class.
                Pass ``DataloaderProfilingConfig()`` to use the default configuration.
            python_profiling_config (PythonProfilingConfig): The configuration for stats
                collected by the Python profiler (cProfile or Pyinstrument).
                Configure it using the
                :class:`~sagemaker.debugger.metrics_config.PythonProfilingConfig` class.
                Pass ``PythonProfilingConfig()`` to use the default configuration.
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The Unix time at which to start profiling.
            duration (float): The duration in seconds to profile.

        .. tip::
            Available profiling range parameter pairs are
            (**start_step** and **num_steps**) and (**start_unix_time** and **duration**).
            The two parameter pairs are mutually exclusive, and this class validates
            if one of the two pairs is used. If both pairs are specified, a
            conflict error occurs.

        """
        self.profiling_parameters = {}
        self._use_default_metrics_configs = False
        self._use_one_config_for_all_metrics = False
        self._use_custom_metrics_configs = False

        self._process_trace_file_parameters(
            local_path, file_max_size, file_close_interval, file_open_fail_threshold
        )
        use_custom_metrics_configs = self._process_metrics_configs(
            detailed_profiling_config,
            dataloader_profiling_config,
            python_profiling_config,
            horovod_profiling_config,
            smdataparallel_profiling_config,
        )

        use_one_config_for_all_metrics = (
            self._process_range_fields(start_step, num_steps, start_unix_time, duration)
            if not use_custom_metrics_configs
            else False
        )

        if not use_custom_metrics_configs and not use_one_config_for_all_metrics:
            self._create_default_metrics_configs()

    def _process_trace_file_parameters(
        self, local_path, file_max_size, file_close_interval, file_open_fail_threshold
    ):
        """Helper function to validate and set the provided trace file parameters.

        Args:
            local_path (str): The path where profiler events have to be saved.
            file_max_size (int): Max size a trace file can be, before being rotated.
            file_close_interval (float): Interval in seconds from the last close, before being
                rotated.
            file_open_fail_threshold (int): Number of times to attempt to open a trace fail before
                marking the writer as unhealthy.

        """
        assert isinstance(local_path, str), ErrorMessages.INVALID_LOCAL_PATH.value
        assert (
            isinstance(file_max_size, int) and file_max_size > 0
        ), ErrorMessages.INVALID_FILE_MAX_SIZE.value
        assert (
            isinstance(file_close_interval, (float, int)) and file_close_interval > 0
        ), ErrorMessages.INVALID_FILE_CLOSE_INTERVAL.value
        assert (
            isinstance(file_open_fail_threshold, int) and file_open_fail_threshold > 0
        ), ErrorMessages.INVALID_FILE_OPEN_FAIL_THRESHOLD.value

        self.profiling_parameters["LocalPath"] = local_path
        self.profiling_parameters["RotateMaxFileSizeInBytes"] = str(file_max_size)
        self.profiling_parameters["RotateFileCloseIntervalInSeconds"] = str(file_close_interval)
        self.profiling_parameters["FileOpenFailThreshold"] = str(file_open_fail_threshold)

    def _process_metrics_configs(self, *metrics_configs):
        """Helper function to validate and set the provided metrics_configs.

        In this case,
        the user specifies configurations for the metrics they want to profile.
        Profiling does not occur
        for metrics if the configurations are not specified for them.

        Args:
            metrics_configs: The list of metrics configs specified by the user.

        Returns:
            bool: Indicates whether custom metrics configs will be used for profiling.

        """
        metrics_configs = [config for config in metrics_configs if config is not None]
        if len(metrics_configs) == 0:
            return False

        for config in metrics_configs:
            config_name = config.name
            config_json = config.to_json_string()
            self.profiling_parameters[config_name] = config_json
        return True

    def _process_range_fields(self, start_step, num_steps, start_unix_time, duration):
        """Helper function to validate and set the provided range fields.

        Profiling occurs
        for all of the metrics using these fields as the specified range and default parameters
        for the rest of the configuration fields (if necessary).

        Args:
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The UNIX time at which to start profiling.
            duration (float): The duration in seconds to profile.

        Returns:
            bool: Indicates whether a custom step or time range will be used for profiling.

        """
        if start_step is num_steps is start_unix_time is duration is None:
            return False

        for config_class in ALL_METRIC_CONFIGS:
            config = config_class(
                start_step=start_step,
                num_steps=num_steps,
                start_unix_time=start_unix_time,
                duration=duration,
            )
            config_name = config.name
            config_json = config.to_json_string()
            self.profiling_parameters[config_name] = config_json
        return True

    def _create_default_metrics_configs(self):
        """Helper function for creating the default configs for each set of metrics."""
        for config_class in ALL_METRIC_CONFIGS:
            config = config_class(profile_default_steps=True)
            config_name = config.name
            config_json = config.to_json_string()
            self.profiling_parameters[config_name] = config_json


def _convert_key_and_value(key, value):
    """Helper function to convert the provided key and value pair (from a dictionary) to a string.

    Args:
        key (str): The key in the dictionary.
        value: The value for this key.

    Returns:
        str: The provided key value pair as a string.

    """
    updated_key = f'"{key}"' if isinstance(key, str) else key
    updated_value = f'"{value}"' if isinstance(value, str) else value

    return f"{updated_key}: {updated_value}, "


def convert_json_config_to_string(config):
    """Helper function to convert the dictionary config to a string.

    Calling eval on this string should result in the original dictionary.

    Args:
        config (dict): The config to be converted to a string.

    Returns:
        str: The config dictionary formatted as a string.

    """
    json_string = "{"
    for key, value in config.items():
        json_string += _convert_key_and_value(key, value)
    json_string += "}"
    return json_string


def is_valid_unix_time(unix_time):
    """Helper function to determine whether the provided UNIX time is valid.

    Args:
        unix_time (int): The user provided UNIX time.

    Returns:
        bool: Indicates whether the provided UNIX time was valid or not.

    """
    try:
        datetime.fromtimestamp(unix_time)
        return True
    except (OverflowError, ValueError):
        return False


def is_valid_regex(regex):
    """Helper function to determine whether the provided regex is valid.

    Args:
        regex (str): The user provided regex.

    Returns:
        bool: Indicates whether the provided regex was valid or not.

    """
    try:
        re.compile(regex)
        return True
    except (re.error, TypeError):
        return False


class ErrorMessages(Enum):
    """Enum to store all possible messages during failures in validation of user arguments."""

    INVALID_LOCAL_PATH = "local_path must be a string!"
    INVALID_FILE_MAX_SIZE = "file_max_size must be an integer greater than 0!"
    INVALID_FILE_CLOSE_INTERVAL = "file_close_interval must be a float/integer greater than 0!"
    INVALID_FILE_OPEN_FAIL_THRESHOLD = "file_open_fail threshold must be an integer greater than 0!"
    INVALID_PROFILE_DEFAULT_STEPS = "profile_default_steps must be a boolean!"
    INVALID_START_STEP = "start_step must be integer greater or equal to 0!"
    INVALID_NUM_STEPS = "num_steps must be integer greater than 0!"
    INVALID_START_UNIX_TIME = "start_unix_time must be valid integer unix time!"
    INVALID_DURATION = "duration must be float greater than 0!"
    FOUND_BOTH_STEP_AND_TIME_FIELDS = (
        "Both step and time fields cannot be specified in the metrics config!"
    )
    INVALID_METRICS_REGEX = "metrics_regex is invalid!"
    INVALID_PYTHON_PROFILER = "python_profiler must be of type PythonProfiler!"
    INVALID_CPROFILE_TIMER = "cprofile_timer must be of type cProfileTimer"


class PythonProfiler(Enum):
    """Enum to list the Python profiler options for Python profiling.

    .. py:attribute:: CPROFILE

        Use to choose ``"cProfile"``.

    .. py:attribute:: PYINSTRUMENT

        Use to choose ``"Pyinstrument"``.

    """

    CPROFILE = "cprofile"
    PYINSTRUMENT = "pyinstrument"


class cProfileTimer(Enum):
    """Enum to list the possible cProfile timers for Python profiling.

    .. py:attribute:: TOTAL_TIME

        Use to choose ``"total_time"``.

    .. py:attribute:: CPU_TIME

        Use to choose ``"cpu_time"``.

    .. py:attribute:: OFF_CPU_TIME

        Use to choose ``"off_cpu_time"``.

    """

    TOTAL_TIME = "total_time"
    CPU_TIME = "cpu_time"
    OFF_CPU_TIME = "off_cpu_time"
    DEFAULT = "default"
