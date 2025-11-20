# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from typing import Any


def dict_to_object_str(dictionary: dict[str, Any], object_name: str) -> str:
    """
    Convert a dictionary to a string representation of an object.

    Example:
        >>> dict_to_object_str({"a": 1, "b": "hello"}, "MyObject")
        'MyObject[a=1, b="hello"]'

    Args:
        dictionary (Dict[str, Any]): Dictionary to convert.
        object_name (str): Name of the object.

    Returns:
        str: String representation of the object.
    """

    def format_item(key):
        value = dictionary[key]
        quote = '"' if isinstance(value, str) else ""
        return f"{key}={quote}{value!s}{quote}"

    if dictionary is None:
        return None

    data = ", ".join(map(format_item, dictionary.keys()))
    return f"{object_name}[{data}]"
