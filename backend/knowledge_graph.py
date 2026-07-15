"""
Knowledge Graph for the Wizarding World.
Tracks all world entities (locations, NPCs, quests, items, factions)
and their relationships. Exports to D3.js-compatible JSON for visualization.
"""
import json
import os
import networkx as nx
from typing import Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class WizardingWorldGraph:
    """
    NetworkX DiGraph representing the full wizarding world state.
    Nodes: locations, npcs, quests, items, factions, player
    Edges: relationships between entities
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._load_world_data()

    def _load_world_data(self):
        """Load world lore and quests into the graph."""
        try:
            with open(os.path.join(DATA_DIR, "world_lore.json"), "r") as f:
                lore = json.load(f)
            with open(os.path.join(DATA_DIR, "quests.json"), "r") as f:
                quest_data = json.load(f)
        except FileNotFoundError:
            lore = {"locations": [], "npcs": [], "factions": [], "items": [], "lore_entries": []}
            quest_data = {"quests": []}

        # Add locations
        for loc in lore.get("locations", []):
            self.graph.add_node(
                loc["id"],
                label=loc["name"],
                type="location",
                description=loc.get("description", ""),
                atmosphere=loc.get("atmosphere", ""),
                secrets=loc.get("secrets", [])
            )
            # Add connections between locations
            for connected in loc.get("connected_to", []):
                self.graph.add_edge(loc["id"], connected, relationship="connected_to")

        # Add NPCs
        for npc in lore.get("npcs", []):
            self.graph.add_node(
                npc["id"],
                label=npc["name"],
                type="npc",
                role=npc.get("role", ""),
                attitude=npc.get("attitude", "neutral"),
                description=npc.get("description", ""),
                knows=npc.get("knows", []),
                quest_giver=npc.get("quest_giver", False),
                alive=True
            )
            # NPC lives in location
            if npc.get("location"):
                self.graph.add_edge(npc["id"], npc["location"], relationship="located_in")

        # Add factions
        for faction in lore.get("factions", []):
            self.graph.add_node(
                faction["id"],
                label=faction["name"],
                type="faction",
                alignment=faction.get("alignment", "neutral"),
                description=faction.get("description", ""),
                player_reputation=faction.get("player_reputation", 0)
            )

        # Add items
        for item in lore.get("items", []):
            self.graph.add_node(
                item["id"],
                label=item["name"],
                type="item",
                description=item.get("description", ""),
                magical=item.get("magical", False),
                discovered=False
            )

        # Add quests
        for quest in quest_data.get("quests", []):
            self.graph.add_node(
                quest["id"],
                label=quest["title"],
                type="quest",
                status=quest.get("status", "locked"),
                difficulty=quest.get("difficulty", "medium"),
                giver=quest.get("giver", ""),
                objectives=[o["text"] for o in quest.get("objectives", [])],
                completed_objectives=[]
            )
            # Quest given by NPC
            if quest.get("giver"):
                self.graph.add_edge(quest["giver"], quest["id"], relationship="gives_quest")
            # Quest chain connections
            for connected_quest in quest.get("connected_quests", []):
                self.graph.add_edge(quest["id"], connected_quest, relationship="leads_to")

        # Add player node
        self.graph.add_node(
            "player",
            label="You",
            type="player",
            location="loc_001",
            house="Unselected"
        )
        self.graph.add_edge("player", "loc_001", relationship="currently_at")

    # ── World State Queries ───────────────────────────────────────────────────

    def get_location_info(self, location_id: str) -> Optional[Dict]:
        """Get full info about a location."""
        if location_id not in self.graph:
            return None
        return dict(self.graph.nodes[location_id])

    def get_npc_info(self, npc_id: str) -> Optional[Dict]:
        """Get full info about an NPC."""
        if npc_id not in self.graph:
            return None
        return dict(self.graph.nodes[npc_id])

    def get_npcs_at_location(self, location_id: str) -> List[Dict]:
        """Get all NPCs currently at a location."""
        npcs = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "npc":
                # Check if NPC is at this location
                for _, target, edge_data in self.graph.out_edges(node, data=True):
                    if target == location_id and edge_data.get("relationship") == "located_in":
                        npcs.append({"id": node, **data})
        return npcs

    def get_active_quests(self) -> List[Dict]:
        """Get all currently active quests."""
        quests = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "quest" and data.get("status") == "active":
                quests.append({"id": node, **data})
        return quests

    def get_available_quests(self) -> List[Dict]:
        """Get all available (not locked) quests."""
        quests = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "quest" and data.get("status") in ["available", "active"]:
                quests.append({"id": node, **data})
        return quests

    def get_location_context(self, location_id: str) -> str:
        """Build a narrative context string for a location."""
        loc = self.get_location_info(location_id)
        if not loc:
            return "An unknown place."

        npcs = self.get_npcs_at_location(location_id)
        npc_names = [n["label"] for n in npcs if n.get("alive", True)]

        context = f"Location: {loc['label']}. {loc.get('description', '')}"
        if npc_names:
            context += f" Present: {', '.join(npc_names)}."
        return context

    # ── World State Updates ───────────────────────────────────────────────────

    def update_player_location(self, location_id: str):
        """Move player to new location."""
        # Remove old location edge
        edges_to_remove = [
            (u, v) for u, v, d in self.graph.out_edges("player", data=True)
            if d.get("relationship") == "currently_at"
        ]
        self.graph.remove_edges_from(edges_to_remove)
        # Add new location edge
        self.graph.nodes["player"]["location"] = location_id
        self.graph.add_edge("player", location_id, relationship="currently_at")
        # Mark location as visited
        if location_id in self.graph:
            self.graph.nodes[location_id]["visited"] = True

    def update_quest_status(self, quest_id: str, status: str):
        """Update a quest's status."""
        if quest_id in self.graph:
            self.graph.nodes[quest_id]["status"] = status
            # Unlock connected quests if this one is completed
            if status == "completed":
                for _, target, edge_data in self.graph.out_edges(quest_id, data=True):
                    if edge_data.get("relationship") == "leads_to":
                        if self.graph.nodes[target].get("status") == "locked":
                            self.graph.nodes[target]["status"] = "available"

    def update_npc_attitude(self, npc_id: str, attitude: str):
        """Update an NPC's attitude toward the player."""
        if npc_id in self.graph:
            self.graph.nodes[npc_id]["attitude"] = attitude

    def mark_npc_met(self, npc_id: str):
        """Record that the player has met this NPC."""
        if npc_id in self.graph:
            self.graph.nodes[npc_id]["player_met"] = True
            self.graph.add_edge("player", npc_id, relationship="knows")

    def discover_item(self, item_id: str):
        """Mark an item as discovered by the player."""
        if item_id in self.graph:
            self.graph.nodes[item_id]["discovered"] = True
            self.graph.add_edge("player", item_id, relationship="discovered")

    def update_faction_reputation(self, faction_id: str, delta: int):
        """Update player's reputation with a faction."""
        faction_map = {
            "faction_001": "Ministry of Magic",
            "faction_002": "Order of the Phoenix",
            "faction_003": "Hollow Circle",
            "faction_004": "Centaur Herd"
        }
        for fid, fname in faction_map.items():
            node_id = None
            for node, data in self.graph.nodes(data=True):
                if data.get("type") == "faction" and (node == faction_id or data.get("label") == faction_id):
                    node_id = node
                    break
            if node_id:
                current = self.graph.nodes[node_id].get("player_reputation", 0)
                self.graph.nodes[node_id]["player_reputation"] = max(-100, min(100, current + delta))

    # ── D3.js Export ─────────────────────────────────────────────────────────

    def to_d3_json(self, player_location: Optional[str] = None) -> Dict:
        """Export graph in D3.js force-directed format."""
        node_colors = {
            "location": "#4A90D9",
            "npc": "#7ED321",
            "quest": "#F5A623",
            "item": "#BD10E0",
            "faction": "#E74C3C",
            "player": "#FFD700"
        }

        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            is_current = node_id == player_location

            nodes.append({
                "id": node_id,
                "label": data.get("label", node_id),
                "type": node_type,
                "color": "#FFD700" if is_current else node_colors.get(node_type, "#999"),
                "size": 20 if is_current else (12 if node_type == "player" else 8),
                "properties": {k: v for k, v in data.items()
                               if k not in ("label", "type") and not isinstance(v, list)},
                "current": is_current
            })

        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "relationship": data.get("relationship", "related_to")
            })

        return {"nodes": nodes, "links": edges}

    def get_world_summary(self) -> str:
        """Get a brief text summary of current world state for the LLM."""
        active_quests = self.get_active_quests()
        quest_names = [q["label"] for q in active_quests]

        player_loc = self.graph.nodes.get("player", {}).get("location", "unknown")
        loc_info = self.get_location_info(player_loc)
        loc_name = loc_info.get("label", "Unknown") if loc_info else "Unknown"

        return (
            f"Player is at: {loc_name}. "
            f"Active quests: {', '.join(quest_names) if quest_names else 'None'}. "
            f"Total locations discovered: {sum(1 for _, d in self.graph.nodes(data=True) if d.get('visited'))}."
        )


# Global instance
world_graph = WizardingWorldGraph()
