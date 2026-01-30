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
        self._entities: Set[int] = set()

    def create_entity(self) -> int:
        entity = self._next_id
        self._next_id += 1
        self._entities.add(entity)
        return entity

    def add_component(self, entity: int, component: Any) -> None:
        comp_type = type(component)
        if comp_type not in self._components:
            self._components[comp_type] = {}

        self._components[comp_type][entity] = component

    def add_components(self, entity: int, *components: Any) -> None:
        for c in components:
            comp_type = type(c)
            if comp_type not in self._components:
                self._components[comp_type] = {}

            self._components[comp_type][entity] = c

    def get_component(self, entity: int, comp_type: Type[T]) -> T | None:
        store = self._components.get(comp_type, None)
        return store.get(entity) if store else None

    def get_components(self, comp_type: Type[T]) -> Dict[int, T]:
        """
        Returns {entity_id: component} dictionary for a type.
        """
        return self._components.get(comp_type, {})

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

        # Get the dictionary for the first component type
        primary_store = self._components.get(comp_types[0], {})

        for entity, comp1 in primary_store.items():
            result = [comp1]
            has_all = True

            # Check if this entity has the other components
            for other_type in comp_types[1:]:
                store = self._components.get(other_type, {})
                if entity not in store:
                    has_all = False
                    break
                result.append(store[entity])

            if has_all:
                yield entity, tuple(result)

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
