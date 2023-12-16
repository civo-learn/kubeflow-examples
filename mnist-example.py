import kfp
import kfp.dsl as dsl
from kfp import components

from kubeflow.katib import ApiClient
from kubeflow.katib import V1beta1ExperimentSpec
from kubeflow.katib import V1beta1AlgorithmSpec
from kubeflow.katib import V1beta1ObjectiveSpec
from kubeflow.katib import V1beta1ParameterSpec
from kubeflow.katib import V1beta1FeasibleSpace
from kubeflow.katib import V1beta1TrialTemplate
from kubeflow.katib import V1beta1TrialParameterSpec


def create_katib_experiment_task(experiment_name, experiment_namespace, training_steps):
    max_trial_count = 5
    max_failed_trial_count = 3
    parallel_trial_count = 2

    objective = V1beta1ObjectiveSpec(
        type="minimize", goal=0.001, objective_metric_name="loss"
    )

    algorithm = V1beta1AlgorithmSpec(
        algorithm_name="random",
    )

    parameters = [
        V1beta1ParameterSpec(
            name="learning_rate",
            parameter_type="double",
            feasible_space=V1beta1FeasibleSpace(min="0.01", max="0.05"),
        ),
        V1beta1ParameterSpec(
            name="batch_size",
            parameter_type="int",
            feasible_space=V1beta1FeasibleSpace(min="80", max="100"),
        ),
    ]

    trial_spec = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "TFJob",
        "spec": {
            "tfReplicaSpecs": {
                "Chief": {
                    "replicas": 1,
                    "restartPolicy": "OnFailure",
                    "template": {
                        "metadata": {
                            "annotations": {"sidecar.istio.io/inject": "false"}
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": "tensorflow",
                                    "image": "docker.io/liuhougangxa/tf-estimator-mnist",
                                    "command": [
                                        "python",
                                        "/opt/model.py",
                                        "--tf-train-steps=" + str(training_steps),
                                        "--tf-learning-rate=${trialParameters.learningRate}",
                                        "--tf-batch-size=${trialParameters.batchSize}",
                                    ],
                                }
                            ]
                        },
                    },
                },
                "Worker": {
                    "replicas": 1,
                    "restartPolicy": "OnFailure",
                    "template": {
                        "metadata": {
                            "annotations": {"sidecar.istio.io/inject": "false"}
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": "tensorflow",
                                    "image": "docker.io/liuhougangxa/tf-estimator-mnist",
                                    "command": [
                                        "python",
                                        "/opt/model.py",
                                        "--tf-train-steps=" + str(training_steps),
                                        "--tf-learning-rate=${trialParameters.learningRate}",
                                        "--tf-batch-size=${trialParameters.batchSize}",
                                    ],
                                }
                            ]
                        },
                    },
                },
            }
        },
    }

    trial_template = V1beta1TrialTemplate(
        primary_container_name="tensorflow",
        trial_parameters=[
            V1beta1TrialParameterSpec(
                name="learningRate",
                description="Learning rate for the training model",
                reference="learning_rate",
            ),
            V1beta1TrialParameterSpec(
                name="batchSize",
                description="Batch size for the model",
                reference="batch_size",
            ),
        ],
        trial_spec=trial_spec,
    )

    experiment_spec = V1beta1ExperimentSpec(
        max_trial_count=max_trial_count,
        max_failed_trial_count=max_failed_trial_count,
        parallel_trial_count=parallel_trial_count,
        objective=objective,
        algorithm=algorithm,
        parameters=parameters,
        trial_template=trial_template,
    )

    katib_experiment_launcher_op = components.load_component_from_url(
        "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/katib-launcher/component.yaml"
    )
    op = katib_experiment_launcher_op(
        experiment_name=experiment_name,
        experiment_namespace=experiment_namespace,
        experiment_spec=ApiClient().sanitize_for_serialization(experiment_spec),
        experiment_timeout_minutes=60,
        delete_finished_experiment=False,
    )

    return op


def convert_katib_results(katib_results) -> str:
    import json
    import pprint

    katib_results_json = json.loads(katib_results)
    print("Katib results:")
    pprint.pprint(katib_results_json)
    best_hps = []
    for pa in katib_results_json["currentOptimalTrial"]["parameterAssignments"]:
        if pa["name"] == "learning_rate":
            best_hps.append("--tf-learning-rate=" + pa["value"])
        elif pa["name"] == "batch_size":
            best_hps.append("--tf-batch-size=" + pa["value"])
    print("Best Hyperparameters: {}".format(best_hps))
    return " ".join(best_hps)


def create_tfjob_task(
    tfjob_name, tfjob_namespace, training_steps, katib_op, model_volume_op
):
    import json

    convert_katib_results_op = components.func_to_container_op(convert_katib_results)
    best_hp_op = convert_katib_results_op(katib_op.output)
    best_hps = str(best_hp_op.output)
    tfjob_chief_spec = {
        "replicas": 1,
        "restartPolicy": "OnFailure",
        "template": {
            "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
            "spec": {
                "containers": [
                    {
                        "name": "tensorflow",
                        "image": "docker.io/liuhougangxa/tf-estimator-mnist",
                        "command": ["sh", "-c"],
                        "args": [
                            "python /opt/model.py --tf-export-dir=/mnt/export --tf-train-steps={} {}".format(
                                training_steps, best_hps
                            )
                        ],
                        "volumeMounts": [
                            {"mountPath": "/mnt/export", "name": "kfaas-docs"}
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "model-volume",
                        "persistentVolumeClaim": {
                            "claimName": str(model_volume_op.outputs["name"])
                        },
                    }
                ],
            },
        },
    }

    tfjob_worker_spec = {
        "replicas": 1,
        "restartPolicy": "OnFailure",
        "template": {
            "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
            "spec": {
                "containers": [
                    {
                        "name": "tensorflow",
                        "image": "docker.io/liuhougangxa/tf-estimator-mnist",
                        "command": [
                            "sh",
                            "-c",
                        ],
                        "args": [
                            "python /opt/model.py --tf-export-dir=/mnt/export --tf-train-steps={} {}".format(
                                training_steps, best_hps
                            )
                        ],
                    }
                ],
            },
        },
    }

    tfjob_launcher_op = components.load_component_from_url(
        "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/launcher/component.yaml"
    )
    op = tfjob_launcher_op(
        name=tfjob_name,
        namespace=tfjob_namespace,
        chief_spec=json.dumps(tfjob_chief_spec),
        worker_spec=json.dumps(tfjob_worker_spec),
        tfjob_timeout_minutes=60,
        delete_finished_tfjob=False,
    )
    return op


def create_serving_task(model_name, model_namespace, tfjob_op, model_volume_op):
    api_version = "serving.kserve.io/v1beta1"
    serving_component_url = "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kserve/component.yaml"
    inference_service = """
apiVersion: "{}"
kind: "InferenceService"
metadata:
  name: {}
  namespace: {}
  annotations:
    "sidecar.istio.io/inject": "false"
spec:
  predictor:
    tensorflow:
      storageUri: "pvc://{}/"
""".format(
        api_version, model_name, model_namespace, str(model_volume_op.outputs["name"])
    )

    serving_launcher_op = components.load_component_from_url(serving_component_url)
    serving_launcher_op(action="apply", inferenceservice_yaml=inference_service).after(
        tfjob_op
    )


name = "kfaas-docs"
namespace = "my-profile"
training_steps = "1"


@dsl.pipeline(
    name="End to End Pipeline",
    description="An end to end mnist example including hyperparameter tuning, train and inference",
)
def mnist_pipeline(name=name, namespace=namespace, training_steps=training_steps):
    katib_op = create_katib_experiment_task(name, namespace, training_steps)

    model_volume_op = dsl.VolumeOp(
        name="model", resource_name="model", size="2Gi", modes=dsl.VOLUME_MODE_RWO
    )

    tfjob_op = create_tfjob_task(
        name, namespace, training_steps, katib_op, model_volume_op
    )
    create_serving_task(name, namespace, tfjob_op, model_volume_op)


kfp.compiler.Compiler().compile(
    pipeline_func=mnist_pipeline, package_path="pipeline.yaml"
)
