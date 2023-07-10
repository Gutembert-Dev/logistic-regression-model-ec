"""Example workflow pipeline script for CustomerChurn pipeline.
                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)
Implements a get_pipeline(**kwargs) method.
"""
# pylint: disable=R0914,C0103,W0613

import os
import datetime
import uuid

import stepfunctions
from stepfunctions import steps
from stepfunctions.inputs import ExecutionInput
from stepfunctions.steps import Chain, ProcessingStep
from stepfunctions.workflow import Workflow

import sagemaker
import sagemaker.session
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn

processing_date = datetime.date.today()


def sagemaker_chain_workflow(
    image_uri,
    source_data_path,
    model_path,
    role=None,
    source_bucket=None,
    preprocess_bucket=None,
    version=None,
):
    """Gets a SageMaker ML Pipeline instance working with on customer data.
    Args:
        image_uri:
        source_data_path:
        model_path:
        role: IAM role to create and run steps and pipeline.
        source_bucket: the bucket to use for storing the artifacts.
        preprocess_bucket: the bucket to use for storing the artifacts.
        model_bucket: the bucket to use for storing the artifacts.
    Returns:
        an instance of a pipeline
    """
    sagemaker_session = sagemaker.session.Session()

    execution_input = ExecutionInput(
        schema={
            "PreprocessingJobName": str,
            "TrainingJobName": str,
            "EvaluationProcessingJobName": str,
            "ModelName": str,
            "EndpointName": str,
        }
    )

    # Generate unique names for Pre-Processing Job, Training Job, and Model Evaluation Job for the Step Functions Workflow
    training_job_name = f"promise-to-pay-training-{uuid.uuid1().hex}"  # Each Training Job requires a unique name
    preprocessing_job_name = f"promise-to-pay-preprocessing-{uuid.uuid1().hex}"  # Each Preprocessing job requires a unique name,

    # Processing step for feature engineering
    script_processor = ScriptProcessor(
        command=["python3"],
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=sagemaker_session,
    )

    inputs = [
        ProcessingInput(
            source=source_data_path,
            destination="/opt/ml/processing/input",
            input_name="input-1",
        )
    ]
    outputs = [
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/train",
            destination=f"s3://{preprocess_bucket}/{processing_date}/{version}",
        ),
        ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/validation",
            destination=f"s3://{preprocess_bucket}/{processing_date}/{version}",
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/test",
            destination=f"s3://{preprocess_bucket}/{processing_date}/{version}",
        ),
    ]

    inputs_evaluation = [
        ProcessingInput(
            source=f"s3://{preprocess_bucket}/{processing_date}/{version}",
            destination="/opt/ml/processing/test",
            input_name="validation",
        ),
        ProcessingInput(
            source=f"{model_path}/{training_job_name}/{'output/model.tar.gz'}",
            destination="/opt/ml/processing/model",
            input_name="model",
        ),
    ]

    outputs_evaluation = [
        ProcessingOutput(
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{preprocess_bucket}/{processing_date}/{version}",
            output_name="evaluation",
        ),
    ]

    processing_step = ProcessingStep(
        "Promise To Pay Pre-processing Step",
        processor=script_processor,
        job_name=execution_input["PreprocessingJobName"],
        inputs=inputs,
        outputs=outputs,
        container_entrypoint=[
            "python3",
            os.path.join("explore_ai_demo", "preprocess.py"),
        ],
    )

    # Training step for generating model artifacts
    sklearn = SKLearn(
        entry_point="explore_ai_demo/train.py",
        output_path=model_path,
        framework_version="0.23-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
        role=role,
        sagemaker_session=sagemaker_session,
    )

    training_step = steps.TrainingStep(
        "Promise To Pay Train Step",
        estimator=sklearn,
        data={
            "train": sagemaker.TrainingInput(
                f"s3://{preprocess_bucket}/{processing_date}/{version}",
                content_type="text/csv",
            )
        },
        job_name=execution_input["TrainingJobName"],
        wait_for_completion=True,
    )

    evaluation_step = ProcessingStep(
        "Promise To Pay Evaluation Step",
        processor=script_processor,
        job_name=execution_input["EvaluationProcessingJobName"],
        inputs=inputs_evaluation,
        outputs=outputs_evaluation,
        container_entrypoint=[
            "python3",
            os.path.join("explore_ai_demo", "evaluate.py"),
        ],
    )

    failed_state_sagemaker_processing_failure = stepfunctions.steps.states.Fail(
        "MLOPs Workflow failed", cause="SageMakerProcessingJobFailed"
    )

    catch_state_processing = stepfunctions.steps.states.Catch(
        error_equals=["States.TaskFailed"],
        next_step=failed_state_sagemaker_processing_failure,
    )

    processing_step.add_catch(catch_state_processing)
    training_step.add_catch(catch_state_processing)

    workflow_graph = Chain(
        [
            processing_step,
            training_step,
            # model_step,
            # endpoint_config_step,
            # endpoint_step,
            evaluation_step,
        ]
    )

    branching_workflow = Workflow(
        name=f"PromiseToPayModelWorkflow-{uuid.uuid1().hex}",
        definition=workflow_graph,
        role=role,
    )

    branching_workflow.create()

    # Execute workflow
    execution = branching_workflow.execute(
        inputs={
            "PreprocessingJobName": preprocessing_job_name,
            # Each pre processing job (SageMaker processing job) requires a unique name,
            "TrainingJobName": training_job_name,  # Each Sagemaker Training job requires a unique name,
        }
    )

    execution.get_output(wait=True)

    # Clean up
    branching_workflow.delete()


if __name__ == "__main__":
    sagemaker_chain_workflow(
        image_uri=os.environ["IMAGE_URI"],
        source_data_path=os.environ["SOURCE_DATA_PATH"],
        model_path=os.environ["MODEL_PATH"],
        role=os.environ["SAGEMAKER_PIPELINE_ROLE_ARN"],
        source_bucket=os.environ["SOURCE_BUCKET"],
        preprocess_bucket=os.environ["PREPROCESS_BUCKET"],
        version=os.environ["BUILD_ID"],
    )
