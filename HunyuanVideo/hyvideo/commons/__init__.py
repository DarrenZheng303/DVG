import os


def get_rank():
    return int(os.environ.get("RANK", "0"))


def rank0_print(message: str) -> None:
    if get_rank() == 0:
        print(message, flush=True)
