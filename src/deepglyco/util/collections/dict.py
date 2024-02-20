__all__ = ["flatten_dict", "chain_item", "chain_get"]

from typing import Any, Dict, Mapping, Optional, Type, TypeVar, overload


def flatten_dict(d: Mapping[str, Any], sep: str = ".") -> Dict[str, Any]:
    def _flatten_dict_gen(d, parent_key, sep):
        for k, v in d.items():
            new_key = str(parent_key) + sep + str(k) if parent_key is not None else k
            if isinstance(v, Mapping):
                yield from _flatten_dict_gen(v, new_key, sep=sep)
            else:
                yield new_key, v

    return dict(_flatten_dict_gen(d, None, sep))


def chain_item(d: Mapping, *keys):
    result = d
    for k in keys:
        if isinstance(result, Mapping) and k in result:
            result = result[k]
        else:
            raise KeyError(keys)
    return result


def chain_get(d: Mapping, *keys, default=None):
    result = d
    for k in keys:
        if isinstance(result, Mapping) and k in result:
            result = result.get(k, default)
        else:
            result = default
            break
    return result


T = TypeVar("T")


def chain_item_typed(d: Mapping, t: Type[T], *keys, allow_convert: bool = False) -> T:
    result = chain_item(d, *keys)
    if not isinstance(result, t):
        if allow_convert:
            result = t(result)
        else:
            raise TypeError(keys, f"{t} expected but {type(result)} found")
    return result

@overload
def chain_get_typed(
    d: Mapping,
    t: Type[T],
    *keys,
    default: T,
    allow_convert: bool = False,
) -> T:
    ...

@overload
def chain_get_typed(
    d: Mapping,
    t: Type[T],
    *keys,
    default: Optional[T] = None,
    allow_convert: bool = False,
) -> Optional[T]:
    ...

def chain_get_typed(
    d: Mapping,
    t: Type[T],
    *keys,
    default: Optional[T] = None,
    allow_convert: bool = False,
) -> Optional[T]:
    result = chain_get(d, *keys, default=default)
    if result is not None and not isinstance(result, t):
        if allow_convert:
            try:
                result = t(result)
            except:
                result = default
        else:
            result = default
    return result


if __name__ == "__main__":
    d = {"a": 1, "c": {"a": 2, "b": {"x": 3, "y": 4, "z": 5}}, "d": [6, 7, 8]}
    print(flatten_dict(d))
    # {'a': 1, 'c.a': 2, 'c.b.x': 3, 'c.b.y': 4, 'c.b.z': 5, 'd': [6, 7, 8]}

    print(chain_item(d, "c", "b", "x"))
    print(chain_get(d, "c", "b", "x"))
    print(chain_get(d, "c", "b", "s", default=None))
    print(chain_item_typed(d, int, "c", "b", "x"))
    print(chain_item_typed(d, int, "d"))
