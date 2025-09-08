from functools import lru_cache
from itertools import chain
from typing import Dict, Any, Tuple

import pandas as pd
from langgraph.func import entrypoint, task

from NL2SQLEvaluator.db_executor import BaseSQLDBExecutor
from NL2SQLEvaluator.db_executor.base_db_executor import db_executor_worker
from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import evaluator_worker
from NL2SQLEvaluator.task_state import MultipleTasks, AvailableDialect


@entrypoint()
def orchestrator_entrypoint(multiple_tasks: MultipleTasks) -> MultipleTasks:
    batch_size = multiple_tasks.batch_size
    # the creation of the engine must be performed before the workers are launched to avoid deadlocks
    for single_task in multiple_tasks.tasks:
        single_task.engine = get_engine(single_task.relative_base_path, AvailableDialect.sqlite)

    # divide create Tasks of length batch_size
    tasks_list = multiple_tasks.tasks
    results = []
    for i in range(0, len(tasks_list), batch_size):
        batch = tasks_list[i:i + batch_size]
        # Process each batch
        batch_tasks = MultipleTasks(tasks=batch, **multiple_tasks.model_dump(exclude={"tasks"}))
        results.append(initialize_and_launch_workers(batch_tasks).result())

    return aggregator(results).result()


@task()
def initialize_and_launch_workers(multiple_tasks: MultipleTasks) -> MultipleTasks:
    completed_tasks = []
    for single_task in multiple_tasks.tasks:
        single_task = db_executor_worker(single_task).result()
        single_task = evaluator_worker(single_task).result()
        completed_tasks.append(single_task)
    return MultipleTasks(tasks=completed_tasks, **multiple_tasks.model_dump(exclude={"tasks"}))


@task()
def aggregator(executed_batch_tasks: list[MultipleTasks]) -> MultipleTasks:
    task_list = [batch.tasks for batch in executed_batch_tasks]
    task_list = list(chain.from_iterable(task_list))
    return MultipleTasks(tasks=task_list, **executed_batch_tasks[0].model_dump(exclude={"tasks"}))


@lru_cache(maxsize=100)
def get_engine(relative_base_path, db_executor: AvailableDialect, *args, **kwargs) -> BaseSQLDBExecutor:
    if db_executor == AvailableDialect.sqlite:
        from NL2SQLEvaluator.db_executor.sqlite_executor import SqliteDBExecutor
        return SqliteDBExecutor.from_uri(relative_base_path, *args, **kwargs)
    else:
        raise ValueError(f"Database executor not supported: {db_executor}. Supported: {list(AvailableDialect)}")