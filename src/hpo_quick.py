import logging
import argparse
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange,
    HyperParameterOptimizer,
    RandomSearch,           # Optuna/BOHB still work; this is the fallback
)

# ────────────────────────────────────────────────────────────────────────────────
# Pick a search backend (leave as in your original script)
# ────────────────────────────────────────────────────────────────────────────────
try:
    from clearml.automation.optuna import OptimizerOptuna  # noqa
    SearchStrategy = OptimizerOptuna
except ImportError:
    try:
        from clearml.automation.hpbandster import OptimizerBOHB  # noqa
        SearchStrategy = OptimizerBOHB
    except ImportError:
        logging.getLogger().warning("No optuna / hpbandster — falling back to Random Search.")
        SearchStrategy = RandomSearch

# ────────────────────────────────────────────────────────────────────────────────
# Callback
# ────────────────────────────────────────────────────────────────────────────────
def job_complete_callback(job_id, objective_value, objective_iteration,
                          job_parameters, top_performance_job_id):
    print(f"Job {job_id} finished | val_acc={objective_value:.4f} "
          f"(iter {objective_iteration}) → params={job_parameters}")
    if job_id == top_performance_job_id:
        print("🥳 New record! This is the current best model.")

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--template_task_id")
parser.add_argument("--run_as_service", action="store_true")
parser.add_argument("--execution_queue", default="default")
args = parser.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# Optimiser task
# ────────────────────────────────────────────────────────────────────────────────
opt_task = Task.init(
    project_name="FashionMNIST-CNN",
    task_name="HPO-sweep-quick",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
)

if not args.template_task_id:          # fallback if user forgot the ID
    args.template_task_id = Task.get_task(
        project_name="FashionMNIST-CNN",
        task_name="baseline-training",
    ).id

# ────────────────────────────────────────────────────────────────────────────────
# Search-space  ← CHANGED
# Keep it tiny: 2 lrs × 1 batch-size × 2 dropouts = 4 combos max
# plus force a single training epoch
# ────────────────────────────────────────────────────────────────────────────────
search_space = [
    DiscreteParameterRange("Args/lr",          values=[1e-3, 3e-4]),
    DiscreteParameterRange("Args/batch_size",  values=[64]),
    DiscreteParameterRange("Args/dropout",     values=[0.25, 0.5]),
    DiscreteParameterRange("Args/epochs",      values=[1]),  # your train.py must expose --epochs
]

# ────────────────────────────────────────────────────────────────────────────────
# HyperParameterOptimizer  ← CHANGED
# total_max_jobs=5 → only five children
# max/min_iteration_per_job=1 → ClearML stops after first reported epoch
# ────────────────────────────────────────────────────────────────────────────────
an_optimizer = HyperParameterOptimizer(
    base_task_id=args.template_task_id,
    hyper_parameters=search_space,
    objective_metric_title="val_acc",
    objective_metric_series="val_acc",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=2,      # keep GPU/CPU usage low
    optimizer_class=SearchStrategy,
    execution_queue=args.execution_queue,
    total_max_jobs=5,                      # ← CHANGED
    pool_period_min=0.5,                   # check queue every 30 s
    time_limit_per_job=10,                 # minutes; here mostly irrelevant
    min_iteration_per_job=1,               # ← CHANGED
    max_iteration_per_job=1,               # ← CHANGED
)

# ────────────────────────────────────────────────────────────────────────────────
if args.run_as_service:
    opt_task.execute_remotely(queue_name="services", exit_process=True)

an_optimizer.set_report_period(0.5)        # report every 30 s
an_optimizer.start(job_complete_callback)
an_optimizer.set_time_limit(in_minutes=30) # safety net
an_optimizer.wait()

print("\nTop experiments:")
for exp in an_optimizer.get_top_experiments(top_k=3):
    print(f" • {exp.id}  val_acc={exp.get_reported_single_value('val_acc'):.4f}")

an_optimizer.stop()
print("Quick HPO run finished ✅")
