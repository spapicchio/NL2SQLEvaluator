import time
from itertools import chain

from langgraph.func import entrypoint, task

from NL2SQLEvaluator.db_executor.base_db_executor import db_executor_worker
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import evaluator_worker
from NL2SQLEvaluator.orchestrator_state import MultipleTasks


@entrypoint()
def orchestrator_entrypoint(multiple_tasks: MultipleTasks) -> MultipleTasks:
    batch_size = multiple_tasks.batch_size
    # the creation of the engine must be performed before the workers are launched to avoid deadlocks
    # divide create Tasks of length batch_size
    tasks_list = multiple_tasks.tasks
    results = []
    logger = get_logger(__name__, level="INFO")
    start_time = time.time()
    for i in range(0, len(tasks_list), batch_size):
        batch = tasks_list[i:i + batch_size]
        # Process each batch
        batch_tasks = MultipleTasks(tasks=batch, **multiple_tasks.model_dump(exclude={"tasks"}))
        results.append(initialize_and_launch_workers(batch_tasks).result())
        end_time = time.time()
        logger.info(
            f"Processed batch {i // batch_size + 1} of {((len(tasks_list) - 1) // batch_size) + 1}, time taken: {end_time - start_time:.2f} seconds, batch_size {len(batch)}"
        )

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
