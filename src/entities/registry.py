from typing import Any, Dict, Set, Type, TypeVar, List, Tuple, Iterator, Union

T = TypeVar('T')


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

    def get_component(self, entity, comp_type):
        store = self._components.get(comp_type, None)
        return store.get(entity) if store else None

    def get_components(self, comp_type: Type[T]) -> Dict[int, T]:
        """
        Returns {entity_id: component} dictionary for a type.
        """
        return self._components.get(comp_type, {})

    def view(self, *comp_types: Type[Any]) -> Iterator[Tuple[int, List[Any]]]:
        """
        Returns an iterator of (entity, [comp1, comp2]) for entities that have all requested components.
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
                yield entity, result

    def get_singleton(
            self, *comp_types: Type[Any]
    ) -> Union[Tuple[int, List[Any]], Tuple[None, List[None]]]:
        return next(self.view(*comp_types), (None, [None] * len(comp_types)))
