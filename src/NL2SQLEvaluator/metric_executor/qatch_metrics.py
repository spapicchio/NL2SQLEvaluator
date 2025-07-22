from langgraph.func import task


@task
def worker_cell_precision(executed_target: list[tuple], executed_predicted: list[tuple]) -> dict[str, float]:
    pass


@task
def worker_cell_recall(executed_target: list[tuple], executed_predicted: list[tuple]) -> float:
    pass


@task
def worker_tuple_cardinality(executed_target: list[tuple], executed_predicted: list[tuple]) -> float:
    pass


@task
def worker_tuple_constraint(executed_target: list[tuple], executed_predicted: list[tuple]) -> float:
    pass


@task
def worker_tuple_order(executed_target: list[tuple], executed_predicted: list[tuple]) -> float:
    pass


@task
def worker_f1_score(executed_target: list[tuple], executed_predicted: list[tuple]) -> float:
    pass
