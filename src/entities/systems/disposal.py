from entities.components.disposal import Disposal
from entities.components.entity_flags import EntityFlags
from entities.registry import Registry


class DisposalSystem:
    @staticmethod
    def update(registry: Registry) -> None:
        r_disposal = registry.get_singleton(Disposal)
        if not r_disposal:
            raise RuntimeError("DisposalSystem is missing a Disposal singleton")
            
        _, (disposal, ) = r_disposal
        if not disposal.entities_to_dispose:
            return

        to_dispose: set[int] = set(disposal.entities_to_dispose)
        stack: list[int] = list(to_dispose)

        while stack:
            current = stack.pop()
            children = registry.get_children(current)

            for child in children:
                if child in to_dispose:
                    continue

                dispose_child = True
                r_flags = registry.get_components(child, EntityFlags)
                if r_flags is not None:
                    (flags, ) = r_flags
                    dispose_child = flags.dispose_alongside_parent

                if dispose_child:
                    to_dispose.add(child)
                    stack.append(child)

        for entity in to_dispose:
            children = registry.get_children(entity)
            entities_to_orphan: set[int] = set()
            
            for child in children:
                if child not in to_dispose:
                    entities_to_orphan.add(child)
            
            for child in entities_to_orphan:
                registry.set_parent(child, None)

            registry.set_parent(entity, None)
            registry.remove_entity(entity)

        disposal.entities_to_dispose.clear()
