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

        self._parents: Dict[int, int] = {}
        self._children: Dict[int, Set[int]] = {}

        self._query_cache: Dict[Tuple[Type[Any], ...], Set[int]] = {}

    def create_entity(self) -> int:
        entity = self._next_id
        self._next_id += 1
        self._entity_components[entity] = {}
        self._children[entity] = set()
        return entity

    # this is entity removal from the perspective of the registry i.e. a bunch of dicts,
    # not logical entity disposal. if used by systems naively, you will get
    # dangling entity IDs.
    def remove_entity(self, entity: int) -> None:
        parent = self._parents.pop(entity, None)
        if parent is not None and parent in self._children:
            self._children[parent].discard(entity)

        for child in self._children.get(entity, set()):
            self._parents.pop(child, None)
        self._children.pop(entity, None)

        components = self._entity_components.pop(entity, None)
        if components is not None:
            for comp_type in components:
                self._components[comp_type].pop(entity, None)

        for cached_set in self._query_cache.values():
            cached_set.discard(entity)

    def add_components(self, entity: int, *components: Any) -> None:
        if entity not in self._entity_components:
            return

        for c in components:
            comp_type = type(c)
            if comp_type not in self._components:
                self._components[comp_type] = {}

            self._components[comp_type][entity] = c
            self._entity_components[entity][comp_type] = c

        entity_comps = self._entity_components[entity]
        for query_types, cached_set in self._query_cache.items():
            if entity not in cached_set and all(qt in entity_comps for qt in query_types):
                cached_set.add(entity)

    def _get_or_build_cache(self, comp_types: Tuple[Type[Any], ...]) -> Set[int]:
        if comp_types in self._query_cache:
            return self._query_cache[comp_types]

        stores = [self._components.get(ct, {}) for ct in comp_types]
        if any(not store for store in stores):
            self._query_cache[comp_types] = set()
            return self._query_cache[comp_types]

        sorted_stores = sorted(stores, key=len)
        common = set(sorted_stores[0].keys())
        for store in sorted_stores[1:]:
            common &= store.keys()

        self._query_cache[comp_types] = common
        return common

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

    def get_components(self, entity: int, *comp_types: Type[Any]) -> Tuple[Any, ...] | None: # type: ignore
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

    def view(self, *comp_types: Type[Any]) -> Iterator[Tuple[int, Tuple[Any, ...]]]: # type: ignore
        """
        Returns an iterator of (entity, (comp1, comp2...)) for entities with all requested components.
        """
        if not comp_types:
            return

        cached_entities = self._get_or_build_cache(comp_types)
        if not cached_entities:
            return

        stores = [self._components[ct] for ct in comp_types]
        count = len(comp_types)

        if count == 1:
            s0 = stores[0]
            for e in cached_entities:
                yield e, (s0[e],)
        elif count == 2:
            s0, s1 = stores[0], stores[1]
            for e in cached_entities:
                yield e, (s0[e], s1[e])
        elif count == 3:
            s0, s1, s2 = stores[0], stores[1], stores[2]
            for e in cached_entities:
                yield e, (s0[e], s1[e], s2[e])
        elif count == 4:
            s0, s1, s2, s3 = stores[0], stores[1], stores[2], stores[3]
            for e in cached_entities:
                yield e, (s0[e], s1[e], s2[e], s3[e])
        elif count == 5:
            s0, s1, s2, s3, s4 = stores[0], stores[1], stores[2], stores[3], stores[4]
            for e in cached_entities:
                yield e, (s0[e], s1[e], s2[e], s3[e], s4[e])
        else:
            # slow fallback path for >5 components
            for e in cached_entities:
                yield e, tuple(store[e] for store in stores)

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

    def get_singleton(self, *comp_types: Type[Any]) -> Tuple[int, Tuple[Any, ...]] | None: # type: ignore
        """
        Returns the first entity with all requested components.
        """
        if not comp_types:
            return None

        cached_entities = self._get_or_build_cache(comp_types)

        try:
            lowest_entity = min(cached_entities)
        except ValueError:
            # catch empty sequences
            return None

        stores = [self._components[ct] for ct in comp_types]

        return lowest_entity, tuple(store[lowest_entity] for store in stores)

    def set_parent(self, child: int, new_parent: int | None) -> None:
        if child == new_parent:
            raise ValueError(f"Cannot parent entity {child} to itself")
        if child not in self._entity_components:
            raise ValueError(f"Cannot parent non-existent entity {child} to parent {new_parent}")
        if new_parent is not None and new_parent not in self._entity_components:
            raise ValueError(f"Cannot parent entity {child} to non-existent parent {new_parent}")

        old_parent = self._parents.get(child)
        if old_parent == new_parent:
            return

        if old_parent is not None:
            self._children[old_parent].discard(child)

        if new_parent is None:
            self._parents.pop(child, None)
        else:
            self._parents[child] = new_parent
            self._children[new_parent].add(child)

    def get_parent(self, entity: int) -> int | None:
        return self._parents.get(entity)

    def get_children(self, entity: int) -> Set[int]:
        return set(self._children.get(entity, set()))
