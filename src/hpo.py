"""
hpo.py — Hyper-parameter optimization for Fashion-MNIST CNN
===========================================================

Usage (locally):
    python src/hpo.py --template_task_id <BASELINE_TASK_ID>

To run remotely as a ClearML *service* (recommended once you trust it):
    python src/hpo.py --template_task_id <BASELINE_TASK_ID> --run_as_service

What this script does
---------------------
• Creates an *Optimizer* task in the ClearML UI.
• Defines the search-space for learning-rate, batch-size and dropout.
• Picks the best available search strategy (Optuna-TPE ➜ BOHB ➜ Random Search).
• Submits up to 30 child training jobs (3 concurrent by default) into the
  specified execution queue (default: "default").
• Stops automatically when the budget is exhausted and prints the top 3 jobs.
"""
import logging
import argparse

from clearml import Task
from clearml.automation import (
    DiscreteParameterRange,
    UniformParameterRange,
    HyperParameterOptimizer,
    RandomSearch,
)

# ---------------------------------------------------------
# Pick the most advanced optimisation backend available
# ---------------------------------------------------------
try:
    from clearml.automation.optuna import OptimizerOptuna  # noqa

    SearchStrategy = OptimizerOptuna
except ImportError:
    try:
        from clearml.automation.hpbandster import OptimizerBOHB  # noqa

        SearchStrategy = OptimizerBOHB
    except ImportError:
        logging.getLogger().warning(
            "Neither 'optuna' nor 'hpbandster' installed — falling back to Random Search.")
        SearchStrategy = RandomSearch


# ---------------------------------------------------------
# Callback: fires whenever a child run finishes
# ---------------------------------------------------------

def job_complete_callback(job_id: str,
                          objective_value: float,
                          objective_iteration: int,
                          job_parameters: dict,
                          top_performance_job_id: str):
    print(f"Job {job_id} finished | val_acc={objective_value:.4f} "
          f"(iter {objective_iteration}) → params={job_parameters}")
    if job_id == top_performance_job_id:
        print("🥳 New record! This is the current best model.")


# ---------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--template_task_id", required=False,
                    help="ID of the baseline training task to clone.")
parser.add_argument("--run_as_service", action="store_true",
                    help="If set, enqueue the optimizer itself in 'services' queue.")
parser.add_argument("--execution_queue", default="default",
                    help="Queue that will execute the child training jobs.")
args = parser.parse_args()

# ---------------------------------------------------------
# Initialise Optimizer task
# ---------------------------------------------------------
opt_task = Task.init(project_name="FashionMNIST-CNN",
                     task_name="HPO-sweep",
                     task_type=Task.TaskTypes.optimizer,
                     reuse_last_task_id=False)

# Fallback: attempt to locate a baseline called "baseline-training" if no ID given
if not args.template_task_id:
    print("No --template_task_id supplied; trying to locate baseline-training task…")
    args.template_task_id = Task.get_task(
        project_name="FashionMNIST-CNN",
        task_name="baseline-training",
    ).id
    print("Found baseline task:", args.template_task_id)

# ---------------------------------------------------------
# Define search-space
# The section/name strings must match parameters visible in the UI.
# 'Args/…' refers to argparse CLI flags. If you add --dropout to train.py,
# it will appear as Args/dropout. Alternatively use Task.connect() there.
# ---------------------------------------------------------
# big search space that we will actually run
# search_space = [
#     DiscreteParameterRange("Args/lr", values=[1e-5, 1e-4, 1e-3]),
#     DiscreteParameterRange("Args/batch_size", values=[32, 64, 128]),
#     UniformParameterRange("Args/dropout", min_value=0.0, max_value=0.5),
# ]

# small search space for testing
search_space = [
    DiscreteParameterRange("Args/lr", values=[1e-4, 1e-3]),
    DiscreteParameterRange("Args/batch_size", values=[32, 64]),
]

# ---------------------------------------------------------
# Instantiate the HPO controller
# ---------------------------------------------------------
an_optimizer = HyperParameterOptimizer(
    base_task_id=args.template_task_id,
    hyper_parameters=search_space,
    objective_metric_title="val_acc",       # sent from train.py → print & scalar
    objective_metric_series="val_acc",
    objective_metric_sign="max",            # we want high accuracy
    max_number_of_concurrent_tasks=3,
    optimizer_class=SearchStrategy,
    execution_queue=args.execution_queue,
    total_max_jobs=30,
    pool_period_min=2.0,   # check queue every 2 min
    time_limit_per_job=30, # minutes per child run (ignored by Optuna/BOHB)
    min_iteration_per_job=1,
    max_iteration_per_job=20,
)

# ---------------------------------------------------------
# Optional: launch as persistent service
# ---------------------------------------------------------
if args.run_as_service:
    opt_task.execute_remotely(queue_name="services", exit_process=True)

# ---------------------------------------------------------
# Kick-off optimisation loop
# ---------------------------------------------------------
an_optimizer.set_report_period(2.0)  # minutes between reports
an_optimizer.start(job_complete_callback=job_complete_callback)

an_optimizer.set_time_limit(in_minutes=120.0)
an_optimizer.wait()                     # blocks until DONE

# ---------------------------------------------------------
# Result summary
# ---------------------------------------------------------
print("Top 3 experiments:")
for exp in an_optimizer.get_top_experiments(top_k=3):
    print(f" • {exp.id}  val_acc={exp.get_reported_single_value('val_acc'):.4f}")

an_optimizer.stop()
print("All done — happy tuning! 🏁")
