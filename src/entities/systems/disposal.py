from entities.components.disposal import Disposal
from entities.registry import Hierarchy, Registry


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
            r_hier = registry.get_components(current, Hierarchy)
            if r_hier is None:
                continue
            (hierarchy, ) = r_hier

            for child in hierarchy.children:
                if child in to_dispose:
                    continue

                r_child_hier = registry.get_components(child, Hierarchy)
                if r_child_hier is None:
                    continue

                (child_hierarchy, ) = r_child_hier
                if child_hierarchy.dispose_alongside_parent:
                    to_dispose.add(child)
                    stack.append(child)

        for entity in to_dispose:
            r_hier = registry.get_components(entity, Hierarchy)
            if r_hier:
                (hierarchy, ) = r_hier
                entities_to_orphan: set[int] = set()
                for child in hierarchy.children:
                    if child not in to_dispose:
                        entities_to_orphan.add(child)
                for child in entities_to_orphan:
                    registry.set_parent(child, None)

            registry.set_parent(entity, None)
            registry.remove_entity(entity)

        disposal.entities_to_dispose.clear()
