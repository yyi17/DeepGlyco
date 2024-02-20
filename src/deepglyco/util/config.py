import os
from typing import Any, Mapping, Optional, Type, TypeVar, Union, cast, overload


from .collections.dict import chain_get, chain_get_typed, chain_item, chain_item_typed
from .io.json import load_json
from .io.yaml import load_yaml


T = TypeVar("T")


class Configurable:
    def __init__(self, configs: Union[str, dict]):
        self.configs = {}

        if isinstance(configs, str):
            ext = os.path.splitext(configs)[1]
            if ext.lower() in (".yaml", ".yml"):
                configs = load_yaml(configs)
            elif ext.lower() == ".json":
                configs = load_json(configs)
            else:
                configs = load_yaml(configs)

        configs = cast(dict, configs)

        self.set_configs(configs)

    def get_configs(self, deep: bool = True):
        r = {}

        if deep:
            for key, value in self.__dict__.items():
                if hasattr(value, "get_configs"):
                    r[key] = value.get_configs(deep=True)

        if self.configs:
            r.update(self.configs)

        return r

    @overload
    def get_config(
        self,
        *name,
        deep: bool = False,
    ) -> Any:
        ...

    @overload
    def get_config(
        self,
        *name,
        deep: bool = False,
        required: bool,
    ) -> Union[Any, None]:
        ...

    @overload
    def get_config(
        self, *name, deep: bool = False, typed: Type[T], allow_convert: bool = False
    ) -> T:
        ...

    @overload
    def get_config(
        self,
        *name,
        deep: bool = False,
        required: bool,
        typed: Type[T],
        allow_convert: bool = False,
    ) -> Optional[T]:
        ...

    def get_config(
        self,
        *name,
        deep: bool = False,
        required: bool = True,
        typed: Optional[Type[T]] = None,
        allow_convert: bool = False,
    ) -> Union[T, Any, None]:
        if deep:
            config = self.get_configs(deep=True)
        else:
            config = self.configs

        if required:
            if typed is not None:
                r = chain_item_typed(config, typed, *name, allow_convert=allow_convert)
            else:
                r = chain_item(config, *name)
        else:
            if typed is not None:
                r = chain_get_typed(config, typed, *name, allow_convert=allow_convert)
            else:
                r = chain_get(config, *name)
        return r

    def set_configs(self, configs: Mapping[str, Any]):
        if not configs:
            return self

        nested_configs = {}
        for key, value in configs.items():
            if isinstance(value, Mapping) and hasattr(
                getattr(self, key, None), "set_configs"
            ):
                nested_configs[key] = value
            else:
                self.configs[key] = value

        for key, value in nested_configs.items():
            getattr(self, key).set_configs(value)

        return self
