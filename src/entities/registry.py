from typing import Any, Dict, Set, Type, TypeVar, Tuple, Iterator, overload
from dataclasses import dataclass, field

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")


# a pretty special component that Registry explicitly works with, so it's defined here.
@dataclass
class Hierarchy:
    parent: int | None = None
    children: Set[int] = field(default_factory=set)
    dispose_alongside_parent: bool = True


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

    # this is entity removal from the perspective of the registry i.e. a bunch of dicts,
    # not logical entity disposal. if used by systems naively, you will get
    # dangling entity IDs.
    def remove_entity(self, entity: int) -> None:
        components = self._entity_components.pop(entity, None)
        if components is not None:
            for comp_type in components:
                self._components[comp_type].pop(entity, None)

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

    def get_singleton(self, *comp_types: Type[Any]) -> Tuple[int, Tuple[Any, ...]] | None: # type: ignore
        """
        Returns the first entity with all requested components.
        """
        return next(self.view(*comp_types), None)

    def set_parent(self, child: int, new_parent: int | None) -> None:
        if child == new_parent:
            raise ValueError(f"Cannot parent entity {child} to itself")
        if child not in self._entity_components:
            raise ValueError(f"Cannot parent non-existent entity {child} to parent {new_parent}")
        if new_parent is not None and new_parent not in self._entity_components:
            raise ValueError(f"Cannot parent entity {child} to non-existent parent {new_parent}")

        r_child = self.get_components(child, Hierarchy)
        if r_child is None:
            child_node = Hierarchy()
            self.add_components(child, child_node)
        else:
            (child_node, ) = r_child

        if child_node.parent is not None:
            r_old_parent = self.get_components(child_node.parent, Hierarchy)
            if r_old_parent is not None:
                (old_parent_hierarchy, ) = r_old_parent
                if child in old_parent_hierarchy.children:
                    old_parent_hierarchy.children.remove(child)

        child_node.parent = new_parent
        if new_parent is not None:
            r_new_parent = self.get_components(new_parent, Hierarchy)
            if r_new_parent is None:
                new_parent_hierarchy = Hierarchy()
                self.add_components(new_parent, new_parent_hierarchy)
            else:
                (new_parent_hierarchy, ) = r_new_parent
            new_parent_hierarchy.children.add(child)
