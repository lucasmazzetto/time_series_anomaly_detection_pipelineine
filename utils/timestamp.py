

def make_timestamps(count: int, start: int = 1700000000) -> list[int]:
    """@brief Generate a list of sequential Unix timestamps.
    
    @param count (int): The number of timestamps to generate.
    @param start (int, optional): The starting Unix timestamp. Defaults to 1700000000.
    
    @returns list[int]: A list of `count` sequential Unix timestamps starting from `start`.
    """
    return list(range(start, start + count))