from typing import Any, Dict, Set, Type, TypeVar, Tuple, Iterator, overload

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")


class Registry:
    def __init__(self) -> None:
        self._next_id: int = 0
        self._components: Dict[Type[Any], Dict[int, Any]] = {}
        self._entity_components: Dict[int, Dict[Type[Any], Any]] = {}

    def create_entity(self) -> int:
        entity = self._next_id
        self._next_id += 1
        self._entity_components[entity] = {}
        return entity

    def remove_entity(self, entity: int) -> None:
        components = self._entity_components.pop(entity, None)
        if components is not None:
            for comp_type in components:
                self._components[comp_type].pop(entity, None)

    def add_component(self, entity: int, component: Any) -> None:
        if entity not in self._entity_components:
            return

        comp_type = type(component)
        if comp_type not in self._components:
            self._components[comp_type] = {}

        self._components[comp_type][entity] = component
        self._entity_components[entity][comp_type] = component

    def add_components(self, entity: int, *components: Any) -> None:
        if entity not in self._entity_components:
            return

        for c in components:
            comp_type = type(c)
            if comp_type not in self._components:
                self._components[comp_type] = {}

            self._components[comp_type][entity] = c
            self._entity_components[entity][comp_type] = c

    @overload
    def get_components(self, entity: int, c1: Type[T1]) -> Tuple[T1] | None:
        ...

    @overload
    def get_components(self, entity: int, c1: Type[T1], c2: Type[T2]) -> Tuple[T1, T2] | None:
        ...

    @overload
    def get_components(self, entity: int, c1: Type[T1], c2: Type[T2], c3: Type[T3]) -> Tuple[T1, T2, T3] | None:
        ...

    @overload
    def get_components(self, entity: int, c1: Type[T1], c2: Type[T2], c3: Type[T3], c4: Type[T4]) -> Tuple[T1, T2, T3, T4] | None:
        ...

    @overload
    def get_components(self, entity: int, c1: Type[T1], c2: Type[T2], c3: Type[T3], c4: Type[T4], c5: Type[T5]) -> Tuple[T1, T2, T3, T4, T5] | None:
        ...

    def get_components(self, entity: int, *comp_types: Type[Any]) -> Tuple[Any, ...] | None:
        if not comp_types:
            return ()
        comps = self._entity_components.get(entity)
        if comps is None:
            return None
        try:
            return tuple(comps[ct] for ct in comp_types)
        except KeyError:
            return None

    def get_all_components(self, entity: int) -> Dict[Type[Any], Any]:
        return self._entity_components.get(entity, {})

    def get_components_of_type(self, comp_type: Type[T]) -> Dict[int, T]:
        return self._components.get(comp_type, {})

    def view_all(self) -> Iterator[Tuple[int, Dict[Type[Any], Any]]]:
        yield from self._entity_components.items()

    @overload
    def view(self, c1: Type[T1]) -> Iterator[Tuple[int, Tuple[T1]]]:
        ...

    @overload
    def view(self, c1: Type[T1], c2: Type[T2]) -> Iterator[Tuple[int, Tuple[T1, T2]]]:
        ...

    @overload
    def view(self, c1: Type[T1], c2: Type[T2], c3: Type[T3]) -> Iterator[Tuple[int, Tuple[T1, T2, T3]]]:
        ...

    @overload
    def view(self, c1: Type[T1], c2: Type[T2], c3: Type[T3], c4: Type[T4]) -> Iterator[
            Tuple[int, Tuple[T1, T2, T3, T4]]]:
        ...

    @overload
    def view(self, c1: Type[T1], c2: Type[T2], c3: Type[T3], c4: Type[T4], c5: Type[T5]) -> Iterator[
            Tuple[int, Tuple[T1, T2, T3, T4, T5]]]:
        ...

    def view(self, *comp_types: Type[Any]) -> Iterator[Tuple[int, Tuple[Any, ...]]]:
        """
        Returns an iterator of (entity, (comp1, comp2...)) for entities with all requested components.
        """
        if not comp_types:
            return

        stores = [self._components.get(ct, {}) for ct in comp_types]
        if any(not store for store in stores):
            return

        sorted_stores = sorted(stores, key=len)

        common_entities = set(sorted_stores[0].keys())
        for store in sorted_stores[1:]:
            common_entities.intersection_update(store.keys())
            if not common_entities:
                return

        for entity in common_entities:
            yield entity, tuple(store[entity] for store in stores)

    @overload
    def get_singleton(self, c1: Type[T1]) -> \
            Tuple[int, Tuple[T1]] | None:
        ...

    @overload
    def get_singleton(self, c1: Type[T1], c2: Type[T2]) -> \
            Tuple[int, Tuple[T1, T2]] | None:
        ...

    @overload
    def get_singleton(self, c1: Type[T1], c2: Type[T2], c3: Type[T3]) -> \
            Tuple[int, Tuple[T1, T2, T3]] | None:
        ...

    @overload
    def get_singleton(self, c1: Type[T1], c2: Type[T2], c3: Type[T3], c4: Type[T4]) -> \
            Tuple[int, Tuple[T1, T2, T3, T4]] | None:
        ...

    @overload
    def get_singleton(self, c1: Type[T1], c2: Type[T2], c3: Type[T3], c4: Type[T4], c5: Type[T5]) -> \
            Tuple[int, Tuple[T1, T2, T3, T4, T5]] | None:
        ...

    def get_singleton(self, *comp_types: Type[Any]) -> Tuple[int, Tuple[Any, ...]] | None:
        """
        Returns the first entity with all requested components.
        """
        return next(self.view(*comp_types), None)
