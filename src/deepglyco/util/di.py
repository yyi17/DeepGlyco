"""
Non-invasive Dependency Injection Container.
It fills given constructors or factory methods
based on their named arguments.

See the demo usage at the end of file.
https://code.activestate.com/recipes/576609-non-invasive-dependency-injection/

Created by Ivo Danihelka on Sun, 11 Jan 2009 (MIT)

Modifications:
- Port to Python 3
- Use inspect.getfullargspec
- Add isclass in _getargspec as inspect.ismethod(<class>.__init__) returns False
- Add support for generic alias
- Add type hints
"""

__all__ = ["Context"]

import itertools
import logging
from typing import (
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T")

NO_DEFAULT = "NO_DEFAULT"


class Context:
    """A depencency injection container.
    It detects the needed dependencies based on arguments of factories.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Creates empty context."""
        self.instances = {}
        self.factories = {}
        self.logger = logger

    @overload
    def register(self, property: str, factory: Callable, *factory_args, **factory_kw):
        ...

    @overload
    def register(self, property: str, factory: object):
        ...

    def register(
        self,
        property: str,
        factory: Union[T, Callable[..., T]],
        *factory_args,
        **factory_kw
    ):
        """Registers factory for the given property name.
        The factory could be a callable or a raw value.
        Arguments of the factory will be searched
        inside the context by their name.

        The factory_args and factory_kw allow
        to specify extra arguments for the factory.
        """
        if (factory_args or factory_kw) and not callable(factory):
            raise ValueError(
                "Only callable factory supports extra args: %s, %s(%s, %s)"
                % (property, factory, factory_args, factory_kw)
            )

        self.factories[property] = factory, factory_args, factory_kw

    def get(self, property: str):
        """Lookups the given property name in context.
        Raises KeyError when no such property is found.
        """
        if property not in self.factories:
            raise KeyError("No factory for: %s", property)

        if property in self.instances:
            return self.instances[property]

        factory_spec = self.factories[property]
        instance = self._instantiate(property, *factory_spec)
        self.instances[property] = instance
        return instance

    def get_all(self) -> Sequence:
        """Returns instances of all properties."""
        return [self.get(name) for name in self.factories.keys()]

    @overload
    def build(self, factory: Callable[..., T], *factory_args, **factory_kw) -> T:
        ...

    @overload
    def build(self, factory: T) -> T:
        ...

    def build(
        self, factory: Union[T, Callable[..., T]], *factory_args, **factory_kw
    ) -> T:
        """Invokes the given factory to build a configured instance."""
        return self._instantiate("", factory, factory_args, factory_kw)

    def _instantiate(
        self,
        name: str,
        factory: Union[T, Callable[..., T]],
        factory_args: Sequence,
        factory_kw: Mapping[str, object],
    ) -> T:
        if not callable(factory):
            if self.logger:
                self.logger.debug("Property %r: %s", name, factory)
            return factory

        kwargs = self._prepare_kwargs(factory, factory_args, factory_kw)
        if self.logger:
            self.logger.debug(
                "Property %r: %s(%s, %s)", name, factory.__name__, factory_args, kwargs
            )
        return factory(*factory_args, **kwargs)

    def _prepare_kwargs(
        self,
        factory: Callable,
        factory_args: Sequence,
        factory_kw: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Returns keyword arguments usable for the given factory.
        The factory_kw could specify explicit keyword values.
        """
        defaults = get_argdefaults(factory, len(factory_args))

        for arg, default in defaults.items():
            if arg in factory_kw:
                continue
            elif arg in self.factories:
                defaults[arg] = self.get(arg)
            elif default is NO_DEFAULT:
                raise KeyError("No factory for arg: %s" % arg)

        defaults.update(factory_kw)
        return defaults


def get_argdefaults(factory: Callable, num_skipped=0) -> Dict[str, object]:
    """Returns dict of (arg_name, default_value) pairs.
    The default_value could be NO_DEFAULT
    when no default was specified.
    """
    args, defaults = _getargspec(factory)

    if defaults is not None:
        num_without_defaults = len(args) - len(defaults)
        default_values = (NO_DEFAULT,) * num_without_defaults + defaults
    else:
        default_values = (NO_DEFAULT,) * len(args)

    return dict(itertools.islice(zip(args, default_values), num_skipped, None))


def _getargspec(factory: Callable) -> Tuple[Sequence[str], Optional[Tuple]]:
    """Describes needed arguments for the given factory.
    Returns tuple (args, defaults) with argument names
    and default values for args tail.
    """
    import inspect
    import typing

    if isinstance(factory, typing._GenericAlias):  # type: ignore
        factory = factory.__origin__

    # isclass = False
    # if inspect.isclass(factory):
    #     isclass = True
    #     factory = factory.__init__

    # #if hasattr(factory, "__wrapped__"):
    # #    factory = factory.__wrapped__

    # logging.debug("Inspecting %r", factory)
    # args, vargs, vkw, defaults, _, _, _ = inspect.getfullargspec(factory)
    # if isclass or inspect.ismethod(factory):
    #     args = args[1:]

    sig = inspect.signature(factory)
    args = []
    posonlyargs = []
    defaults = ()
    for param in sig.parameters.values():
        kind = param.kind
        name = param.name

        if kind is inspect._ParameterKind.POSITIONAL_ONLY:
            posonlyargs.append(name)
            if param.default is not param.empty:
                defaults += (param.default,)
        elif kind is inspect._ParameterKind.POSITIONAL_OR_KEYWORD:
            args.append(name)
            if param.default is not param.empty:
                defaults += (param.default,)

    if not defaults:
        defaults = None

    return posonlyargs + args, defaults


if __name__ == "__main__":

    class Demo:
        def __init__(self, title, user, console):
            self.title = title
            self.user = user
            self.console = console

        def say_hello(self):
            self.console.println("*** IoC Demo ***")
            self.console.println(self.title)
            self.console.println("Hello %s" % self.user)

    class Console:
        def __init__(self, prefix=""):
            self.prefix = prefix

        def println(self, message):
            print(self.prefix, message)

    ctx = Context()
    ctx.register("user", "some user")
    ctx.register("console", Console, "-->")
    demo = ctx.build(Demo, title="Inversion of Control")
    demo.say_hello()
