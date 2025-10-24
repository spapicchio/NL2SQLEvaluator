from typing import Protocol, TypedDict, Literal


# ---- Types ----
class ChatTurn(TypedDict):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str


ChatMessageHF = list[ChatTurn]  # one conversation


class DataInput(TypedDict):
    db_file: str
    target_query: list[str]
    input_seq: ChatMessageHF


class DataReaderProtocol(Protocol):
    @staticmethod
    def read(file_path: str, *args, **kwargs) -> list[DataInput]:
        ...


def enforce_input_type(input_dict: list[dict]):
    for data in input_dict:
        if 'db_file' not in data or 'target_query' not in data or 'input_seq' not in data:
            raise ValueError("Each input data must contain at least 'db_file', 'target_query', and 'input_seq' keys.")


def read_data_from_file(reader: DataReaderProtocol, file_path: str, *args, **kwargs) -> list[DataInput]:
    data = reader.read(file_path, *args, **kwargs)
    enforce_input_type(data)
    return data
