"""
Jev-Based Distributed Consensus Simulation for Heterogeneous Multi-Agent Robotics

Architecture:
  Traditional consensus (Raft/PBFT): all nodes converge on single global state.
  Jev consensus: each agent proposes local state transitions → Jev validates
  heuristically via choice/noul/score → high-confidence proposals auto-execute,
  low-confidence escalate for group negotiation. Consensus emerges from
  individually validated decisions, not coordinated agreement.

Decision primitives (mirrors Jev API):
  - jev_choice(state, criteria) → pick best from bounded options + confidence
  - jev_noul(state, instructions)  → boolean gate + confidence
  - jev_score(state, criteria[])   → multi-criteria rubric score + confidence
  Gate: conf ≥ 0.75 → auto | else → escalate

Agent types:
  - Drone:         aerial, speed=3, payload=1, battery depletes, comm_range=8
  - GroundCarrier:  surface, speed=1, payload=4, unlimited power, comm_range=5
  - RelayStation:   fixed, comm_range=15, amplifies consensus signals
"""

import json
import math
import random
import heapq
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

random.seed(42)

# ── World Constants ──────────────────────────────────────────────────────────

GRID_W, GRID_H = 40, 40
SIM_TICKS = 120
CONFIDENCE_GATE = 0.75  # Jev gating threshold


# ── Enums ────────────────────────────────────────────────────────────────────

class AgentType(str, Enum):
    DRONE = "drone"
    GROUND = "ground_carrier"
    RELAY = "relay_station"

class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_TRANSIT = "in_transit"
    HANDED_OVER = "handed_over"
    DELIVERED = "delivered"

class DecisionType(str, Enum):
    PATH_CHOICE = "path_choice"
    HANDOVER_GATE = "handover_gate"
    CONFLICT_RESOLVE = "conflict_resolve"
    TASK_ASSIGN = "task_assign"
    STATE_VALIDATE = "state_validate"


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Vec2:
    x: int
    y: int

    def dist(self, other: "Vec2") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return isinstance(other, Vec2) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def to_tuple(self):
        return (self.x, self.y)


@dataclass
class TransportTask:
    id: int
    pickup: Vec2
    dropoff: Vec2
    weight: int  # 1-4
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    carrier_chain: list = field(default_factory=list)  # agent ids in handover chain


@dataclass
class JevDecision:
    """Record of a Jev consensus decision."""
    tick: int
    agent_id: str
    decision_type: str
    state_summary: str
    options: list
    chosen: str
    confidence: float
    gated: str  # "auto" or "escalate"
    outcome: str


# ── Jev Consensus Engine (local, mirrors Jev API surface) ────────────────────

class JevConsensus:
    """
    Local heuristic validator mirroring Jev's typed-decision API.
    Each method returns (decision, confidence, gated_action).
    Swap these calls for live jev_choice / jev_noul / jev_score
    when the TypeSafe plugin is connected.
    """

    def __init__(self, gate_threshold: float = CONFIDENCE_GATE):
        self.gate = gate_threshold
        self.log: list[JevDecision] = []

    def _gate(self, conf: float) -> str:
        return "auto" if conf >= self.gate else "escalate"

    def choice(self, state: dict, criteria: dict[str, str]) -> dict:
        """
        jev_choice equivalent: pick best option from bounded set.
        criteria = {"option_label": "description sentence (>12 chars)"}

        Heuristic: score each option by matching state features against
        description keywords, with noise to simulate model uncertainty.
        """
        scores = {}
        for label, desc in criteria.items():
            # Base score from state-description feature overlap
            state_str = json.dumps(state).lower()
            desc_words = set(desc.lower().split())
            overlap = sum(1 for w in desc_words if w in state_str)
            base = overlap / max(len(desc_words), 1)

            # Domain heuristics
            if "short" in desc.lower() or "direct" in desc.lower():
                base += 0.15
            if "safe" in desc.lower() or "avoid" in desc.lower():
                base += 0.1
            if "fast" in desc.lower():
                base += 0.05

            # Noise to simulate model uncertainty
            scores[label] = max(0, min(1, base + random.gauss(0, 0.12)))

        total = sum(scores.values()) or 1
        probs = {k: v / total for k, v in scores.items()}
        chosen = max(probs, key=probs.get)
        conf = probs[chosen]

        return {
            "choice": chosen,
            "probabilities": probs,
            "confidence": round(conf, 3),
            "gated": self._gate(conf),
        }

    def noul(self, state: dict, instructions: str,
             true_desc: str = "yes", false_desc: str = "no") -> dict:
        """
        jev_noul equivalent: boolean gate decision.
        Used for handover acceptance, collision checks, safety gates.
        """
        state_str = json.dumps(state).lower()
        instr_words = set(instructions.lower().split())
        relevance = sum(1 for w in instr_words if w in state_str)
        base = relevance / max(len(instr_words), 1)

        # Domain-specific adjustments
        if "safe" in instructions.lower():
            # Safety checks tend toward conservative (true = proceed)
            if state.get("battery", 100) > 20 and state.get("distance", 0) < 15:
                base += 0.3
            else:
                base -= 0.2

        if "handover" in instructions.lower():
            # Handover feasibility depends on proximity and payload match
            if state.get("proximity", 99) < 3 and state.get("payload_match", False):
                base += 0.35
            else:
                base -= 0.15

        conf = max(0.1, min(0.99, base + random.gauss(0, 0.08)))
        is_yes = conf > 0.5

        return {
            "noul": true_desc if is_yes else false_desc,
            "is_yes": is_yes,
            "confidence": round(conf, 3),
            "gated": self._gate(conf),
        }

    def score(self, state: dict, criteria: list[dict]) -> dict:
        """
        jev_score equivalent: multi-criteria rubric scoring.
        criteria = [{"name": "safety", "weight": 0.4, "desc": "..."}]
        """
        scores = {}
        weighted_sum = 0
        total_weight = 0

        for c in criteria:
            name = c["name"]
            weight = c.get("weight", 1.0)
            desc = c.get("desc", "")

            # Heuristic scoring per criterion
            base = 0.5
            if name == "safety":
                battery = state.get("battery", 100)
                base = min(1.0, battery / 100 * 0.7 + 0.3)
            elif name == "efficiency":
                dist = state.get("remaining_distance", 10)
                base = max(0.2, 1.0 - dist / 40)
            elif name == "progress":
                progress = state.get("task_progress", 0)
                base = progress
            elif name == "collision_risk":
                nearby = state.get("nearby_agents", 0)
                base = max(0.1, 1.0 - nearby * 0.25)

            score = max(0, min(1, base + random.gauss(0, 0.06)))
            scores[name] = round(score, 3)
            weighted_sum += score * weight
            total_weight += weight

        composite = weighted_sum / max(total_weight, 0.01)
        conf = max(0.1, min(0.99, composite))

        return {
            "score": round(composite, 3),
            "scores": scores,
            "confidence": round(conf, 3),
            "gated": self._gate(conf),
            "legend": {c["name"]: c.get("desc", "") for c in criteria},
        }

    def record(self, tick, agent_id, dtype, state_summary, options, chosen, conf, gated, outcome):
        self.log.append(JevDecision(
            tick=tick, agent_id=agent_id, decision_type=dtype,
            state_summary=state_summary, options=options,
            chosen=chosen, confidence=conf, gated=gated, outcome=outcome
        ))


# ── Pathfinding ──────────────────────────────────────────────────────────────

def astar(grid, start: Vec2, goal: Vec2, can_fly: bool = False) -> list[Vec2]:
    """A* pathfinding. Drones can fly over obstacles."""
    if start == goal:
        return [start]

    open_set = [(0, start.x, start.y)]
    came_from = {}
    g_score = {start.to_tuple(): 0}
    closed = set()

    while open_set:
        _, cx, cy = heapq.heappop(open_set)
        current = (cx, cy)
        if current in closed:
            continue
        closed.add(current)

        if current == goal.to_tuple():
            path = []
            while current in came_from:
                path.append(Vec2(current[0], current[1]))
                current = came_from[current]
            path.append(start)
            return list(reversed(path))

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if not can_fly and grid[ny][nx] == 1:
                    continue
                move_cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                ng = g_score[current] + move_cost
                nkey = (nx, ny)
                if nkey not in g_score or ng < g_score[nkey]:
                    g_score[nkey] = ng
                    f = ng + math.hypot(nx - goal.x, ny - goal.y)
                    heapq.heappush(open_set, (f, nx, ny))
                    came_from[nkey] = current

    return []  # no path found


def find_alternative_paths(grid, start: Vec2, goal: Vec2, can_fly: bool, count: int = 3) -> list[list[Vec2]]:
    """Generate multiple path alternatives by adding waypoint detours."""
    paths = []
    direct = astar(grid, start, goal, can_fly)
    if direct:
        paths.append(direct)

    # Generate alternatives via waypoints offset from midpoint
    mid_x = (start.x + goal.x) // 2
    mid_y = (start.y + goal.y) // 2
    offsets = [(5, 3), (-4, 5), (3, -6), (-5, -3)]

    for ox, oy in offsets:
        if len(paths) >= count:
            break
        wx = max(0, min(GRID_W - 1, mid_x + ox))
        wy = max(0, min(GRID_H - 1, mid_y + oy))
        wp = Vec2(wx, wy)
        if not can_fly and grid[wy][wx] == 1:
            continue
        seg1 = astar(grid, start, wp, can_fly)
        seg2 = astar(grid, wp, goal, can_fly)
        if seg1 and seg2:
            paths.append(seg1 + seg2[1:])

    return paths if paths else [[start]]


# ── Agents ───────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, id: str, agent_type: AgentType, pos: Vec2, **kwargs):
        self.id = id
        self.agent_type = agent_type
        self.pos = pos
        self.speed = kwargs.get("speed", 1)
        self.payload_cap = kwargs.get("payload_cap", 0)
        self.comm_range = kwargs.get("comm_range", 5)
        self.battery = kwargs.get("battery", 100.0)
        self.carrying: Optional[TransportTask] = None
        self.path: list[Vec2] = []
        self.path_idx = 0
        self.state = "idle"  # idle, moving, waiting_handover, delivering
        self.proposed_action = None
        self.history: list[dict] = []

    def can_fly(self) -> bool:
        return self.agent_type == AgentType.DRONE

    def in_range(self, other: "Agent") -> bool:
        return self.pos.dist(other.pos) <= self.comm_range

    def to_state_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.agent_type.value,
            "pos": self.pos.to_tuple(),
            "battery": round(self.battery, 1),
            "carrying": self.carrying.id if self.carrying else None,
            "state": self.state,
            "speed": self.speed,
            "payload_cap": self.payload_cap,
        }

    def snapshot(self, tick):
        return {
            "tick": tick,
            "id": self.id,
            "type": self.agent_type.value,
            "x": self.pos.x,
            "y": self.pos.y,
            "battery": round(self.battery, 1),
            "state": self.state,
            "carrying": self.carrying.id if self.carrying else None,
            "path": [(p.x, p.y) for p in self.path[self.path_idx:]],
        }


# ── World ────────────────────────────────────────────────────────────────────

class World:
    def __init__(self):
        self.grid = [[0] * GRID_W for _ in range(GRID_H)]
        self.agents: dict[str, Agent] = {}
        self.tasks: list[TransportTask] = []
        self.jev = JevConsensus()
        self.tick = 0
        self.timeline: list[dict] = []  # full simulation log

        self._build_obstacles()
        self._spawn_agents()
        self._create_tasks()

    def _build_obstacles(self):
        """Create buildings/no-fly zones on the grid."""
        obstacle_regions = [
            (8, 8, 5, 4), (20, 5, 6, 3), (30, 12, 4, 6),
            (5, 20, 3, 8), (15, 18, 5, 5), (25, 25, 6, 4),
            (10, 32, 4, 3), (33, 30, 5, 5), (18, 35, 3, 3),
        ]
        for rx, ry, rw, rh in obstacle_regions:
            for y in range(ry, min(ry + rh, GRID_H)):
                for x in range(rx, min(rx + rw, GRID_W)):
                    self.grid[y][x] = 1

    def _spawn_agents(self):
        # Drones: fast aerial scouts/carriers
        for i in range(3):
            pos = [Vec2(2, 2), Vec2(38, 2), Vec2(20, 38)][i]
            self.agents[f"drone_{i}"] = Agent(
                f"drone_{i}", AgentType.DRONE, pos,
                speed=3, payload_cap=1, comm_range=8, battery=100
            )

        # Ground carriers: slow heavy haulers
        for i in range(2):
            pos = [Vec2(5, 15), Vec2(35, 35)][i]
            self.agents[f"ground_{i}"] = Agent(
                f"ground_{i}", AgentType.GROUND, pos,
                speed=1, payload_cap=4, comm_range=5, battery=999
            )

        # Relay stations: fixed comm amplifiers
        for i in range(2):
            pos = [Vec2(20, 20), Vec2(10, 30)][i]
            self.agents[f"relay_{i}"] = Agent(
                f"relay_{i}", AgentType.RELAY, pos,
                speed=0, payload_cap=0, comm_range=15, battery=999
            )

    def _create_tasks(self):
        task_defs = [
            (Vec2(3, 3),   Vec2(36, 36), 1),
            (Vec2(37, 3),  Vec2(5, 37),  2),
            (Vec2(2, 35),  Vec2(38, 5),  1),
            (Vec2(30, 8),  Vec2(5, 30),  3),
            (Vec2(15, 5),  Vec2(25, 35), 1),
        ]
        for i, (pickup, dropoff, weight) in enumerate(task_defs):
            self.tasks.append(TransportTask(i, pickup, dropoff, weight))

    # ── Jev Consensus Decisions ──────────────────────────────────────────

    def _jev_path_selection(self, agent: Agent, goal: Vec2) -> list[Vec2]:
        """Use jev_choice to select among alternative paths."""
        alternatives = find_alternative_paths(
            self.grid, agent.pos, goal, agent.can_fly(), count=3
        )
        if len(alternatives) <= 1:
            return alternatives[0] if alternatives else []

        # Build criteria for Jev choice
        criteria = {}
        for i, path in enumerate(alternatives):
            dist = sum(
                path[j].dist(path[j+1]) for j in range(len(path)-1)
            )
            label = f"route_{i}"
            if i == 0:
                criteria[label] = f"Direct route, distance {dist:.0f} units, shortest path"
            else:
                criteria[label] = f"Alternative route via detour, distance {dist:.0f} units, avoids congestion"

        state = {
            **agent.to_state_dict(),
            "goal": goal.to_tuple(),
            "nearby_agents": self._count_nearby(agent),
            "remaining_distance": agent.pos.dist(goal),
        }

        result = self.jev.choice(state, criteria)
        chosen_idx = int(result["choice"].split("_")[1])

        self.jev.record(
            self.tick, agent.id, DecisionType.PATH_CHOICE.value,
            f"{agent.id} selecting path to {goal.to_tuple()}",
            list(criteria.keys()), result["choice"],
            result["confidence"], result["gated"],
            f"Selected route_{chosen_idx} (len={len(alternatives[chosen_idx])})"
        )

        return alternatives[chosen_idx]

    def _jev_handover_check(self, giver: Agent, receiver: Agent, task: TransportTask) -> bool:
        """Use jev_noul to gate load handover between agents."""
        state = {
            "giver": giver.to_state_dict(),
            "receiver": receiver.to_state_dict(),
            "task_weight": task.weight,
            "proximity": giver.pos.dist(receiver.pos),
            "payload_match": receiver.payload_cap >= task.weight,
            "battery": giver.battery,
            "distance": giver.pos.dist(receiver.pos),
        }

        result = self.jev.noul(
            state,
            f"Is handover safe? Giver {giver.id} at battery {giver.battery:.0f}%, "
            f"receiver {receiver.id} has capacity {receiver.payload_cap}, "
            f"proximity {giver.pos.dist(receiver.pos):.1f}, "
            f"task weight {task.weight}",
            true_desc="approve_handover",
            false_desc="reject_handover",
        )

        approved = result["is_yes"] and result["gated"] == "auto"

        # If escalated, apply stricter check (simulates group negotiation)
        if result["gated"] == "escalate":
            approved = (
                giver.pos.dist(receiver.pos) < 2.5
                and receiver.payload_cap >= task.weight
                and receiver.carrying is None
            )

        self.jev.record(
            self.tick, giver.id, DecisionType.HANDOVER_GATE.value,
            f"Handover {task.id} from {giver.id} → {receiver.id}",
            ["approve_handover", "reject_handover"],
            "approve" if approved else "reject",
            result["confidence"], result["gated"],
            f"{'Approved' if approved else 'Rejected'} handover"
        )

        return approved

    def _jev_state_validation(self, agent: Agent) -> dict:
        """Use jev_score to validate an agent's current state."""
        state = {
            **agent.to_state_dict(),
            "remaining_distance": (
                agent.pos.dist(agent.path[-1]) if agent.path and agent.path_idx < len(agent.path) else 0
            ),
            "task_progress": self._task_progress(agent),
            "nearby_agents": self._count_nearby(agent),
        }

        criteria = [
            {"name": "safety", "weight": 0.35, "desc": "Battery and collision safety margin"},
            {"name": "efficiency", "weight": 0.25, "desc": "Distance-to-goal optimality"},
            {"name": "progress", "weight": 0.25, "desc": "Progress toward task completion"},
            {"name": "collision_risk", "weight": 0.15, "desc": "Risk of path conflict with nearby agents"},
        ]

        result = self.jev.score(state, criteria)

        self.jev.record(
            self.tick, agent.id, DecisionType.STATE_VALIDATE.value,
            f"{agent.id} state validation (battery={agent.battery:.0f}%)",
            [c["name"] for c in criteria],
            f"composite={result['score']:.2f}",
            result["confidence"], result["gated"],
            f"Scores: {result['scores']}"
        )

        return result

    def _jev_task_assignment(self, task: TransportTask) -> Optional[str]:
        """Use jev_choice to assign a task to the best available agent."""
        available = [
            a for a in self.agents.values()
            if a.agent_type != AgentType.RELAY
            and a.state == "idle"
            and a.payload_cap >= task.weight
            and (a.agent_type != AgentType.DRONE or a.battery > 30)
        ]
        if not available:
            return None

        criteria = {}
        for a in available:
            dist = a.pos.dist(task.pickup)
            criteria[a.id] = (
                f"{a.agent_type.value} at distance {dist:.0f}, "
                f"capacity {a.payload_cap}, battery {a.battery:.0f}%, "
                f"speed {a.speed}"
            )

        state = {
            "task_id": task.id,
            "pickup": task.pickup.to_tuple(),
            "dropoff": task.dropoff.to_tuple(),
            "weight": task.weight,
            "available_agents": [a.id for a in available],
        }

        result = self.jev.choice(state, criteria)

        if result["gated"] == "auto":
            chosen = result["choice"]
        else:
            # Escalation: pick closest agent deterministically
            chosen = min(available, key=lambda a: a.pos.dist(task.pickup)).id

        self.jev.record(
            self.tick, chosen, DecisionType.TASK_ASSIGN.value,
            f"Assign task {task.id} (weight={task.weight})",
            list(criteria.keys()), chosen,
            result["confidence"], result["gated"],
            f"Assigned to {chosen}"
        )

        return chosen

    def _jev_conflict_resolve(self, a1: Agent, a2: Agent) -> str:
        """Use jev_choice to resolve path conflicts between two agents."""
        criteria = {
            f"{a1.id}_yields": f"{a1.id} pauses and yields path to {a2.id}, battery {a1.battery:.0f}%",
            f"{a2.id}_yields": f"{a2.id} pauses and yields path to {a1.id}, battery {a2.battery:.0f}%",
            "both_reroute": "Both agents compute new paths to avoid each other",
        }

        state = {
            "agent_1": a1.to_state_dict(),
            "agent_2": a2.to_state_dict(),
            "distance": a1.pos.dist(a2.pos),
        }

        result = self.jev.choice(state, criteria)

        self.jev.record(
            self.tick, f"{a1.id}+{a2.id}",
            DecisionType.CONFLICT_RESOLVE.value,
            f"Path conflict between {a1.id} and {a2.id}",
            list(criteria.keys()), result["choice"],
            result["confidence"], result["gated"],
            f"Resolution: {result['choice']}"
        )

        return result["choice"]

    # ── Helpers ───────────────────────────────────────────────────────────

    def _count_nearby(self, agent: Agent) -> int:
        return sum(
            1 for a in self.agents.values()
            if a.id != agent.id and a.pos.dist(agent.pos) < agent.comm_range
        )

    def _task_progress(self, agent: Agent) -> float:
        if not agent.carrying:
            return 0.0
        task = agent.carrying
        total = task.pickup.dist(task.dropoff)
        remaining = agent.pos.dist(task.dropoff)
        return max(0, 1.0 - remaining / max(total, 1))

    def _find_handover_partner(self, agent: Agent) -> Optional[Agent]:
        """Find a nearby agent that can take over a load."""
        if not agent.carrying:
            return None
        for other in self.agents.values():
            if (other.id != agent.id
                and other.agent_type != AgentType.RELAY
                and other.state == "idle"
                and other.carrying is None
                and other.payload_cap >= agent.carrying.weight
                and agent.in_range(other)):
                return other
        return None

    # ── Simulation Loop ──────────────────────────────────────────────────

    def step(self):
        self.tick += 1
        tick_events = {"tick": self.tick, "agents": [], "decisions": [], "events": []}

        # Phase 1: Task assignment via Jev
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                chosen_id = self._jev_task_assignment(task)
                if chosen_id:
                    agent = self.agents[chosen_id]
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_to = chosen_id
                    task.carrier_chain.append(chosen_id)
                    agent.state = "moving"
                    agent.path = self._jev_path_selection(agent, task.pickup)
                    agent.path_idx = 0
                    tick_events["events"].append({
                        "type": "task_assigned",
                        "task": task.id,
                        "agent": chosen_id,
                    })

        # Phase 2: Agent actions
        for agent in self.agents.values():
            if agent.agent_type == AgentType.RELAY:
                tick_events["agents"].append(agent.snapshot(self.tick))
                continue

            # Validate state periodically
            if self.tick % 5 == 0 and agent.state != "idle":
                validation = self._jev_state_validation(agent)
                if validation["gated"] == "escalate" and validation["score"] < 0.4:
                    # Low-confidence state: re-plan
                    tick_events["events"].append({
                        "type": "state_replan",
                        "agent": agent.id,
                        "score": validation["score"],
                    })
                    if agent.carrying and agent.path:
                        agent.path = self._jev_path_selection(agent, agent.path[-1])
                        agent.path_idx = 0

            # Movement
            if agent.state == "moving" and agent.path and agent.path_idx < len(agent.path):
                steps = min(agent.speed, len(agent.path) - agent.path_idx)
                for _ in range(steps):
                    if agent.path_idx < len(agent.path):
                        agent.pos = agent.path[agent.path_idx]
                        agent.path_idx += 1

                # Battery drain for drones
                if agent.agent_type == AgentType.DRONE:
                    agent.battery -= 0.8 * steps

                # Check if reached end of path
                if agent.path_idx >= len(agent.path):
                    # Check what we're at
                    for task in self.tasks:
                        if task.assigned_to == agent.id:
                            if task.status == TaskStatus.ASSIGNED and agent.pos == task.pickup:
                                # Pick up
                                agent.carrying = task
                                task.status = TaskStatus.IN_TRANSIT
                                agent.path = self._jev_path_selection(agent, task.dropoff)
                                agent.path_idx = 0
                                tick_events["events"].append({
                                    "type": "pickup",
                                    "agent": agent.id,
                                    "task": task.id,
                                })
                            elif task.status == TaskStatus.IN_TRANSIT and agent.pos.dist(task.dropoff) < 2:
                                # Deliver
                                task.status = TaskStatus.DELIVERED
                                agent.carrying = None
                                agent.state = "idle"
                                agent.path = []
                                tick_events["events"].append({
                                    "type": "delivery",
                                    "agent": agent.id,
                                    "task": task.id,
                                })

            # Handover logic: drones with low battery try to hand off
            if (agent.agent_type == AgentType.DRONE
                and agent.carrying
                and agent.battery < 40):
                partner = self._find_handover_partner(agent)
                if partner:
                    approved = self._jev_handover_check(agent, partner, agent.carrying)
                    if approved:
                        task = agent.carrying
                        task.carrier_chain.append(partner.id)
                        task.assigned_to = partner.id
                        task.status = TaskStatus.HANDED_OVER
                        partner.carrying = task
                        partner.state = "moving"
                        partner.path = self._jev_path_selection(partner, task.dropoff)
                        partner.path_idx = 0
                        agent.carrying = None
                        agent.state = "idle"
                        agent.path = []
                        tick_events["events"].append({
                            "type": "handover",
                            "from": agent.id,
                            "to": partner.id,
                            "task": task.id,
                        })

            # Conflict detection
            for other in self.agents.values():
                if (other.id > agent.id  # avoid double-check
                    and other.agent_type != AgentType.RELAY
                    and agent.pos.dist(other.pos) < 1.5
                    and agent.state == "moving"
                    and other.state == "moving"):
                    resolution = self._jev_conflict_resolve(agent, other)
                    if "yields" in resolution and resolution != "both_reroute":
                        yielder_id = resolution.rsplit("_yields", 1)[0]
                        if yielder_id in self.agents:
                            yielder = self.agents[yielder_id]
                            yielder.path_idx = max(0, yielder.path_idx - 1)
                    tick_events["events"].append({
                        "type": "conflict",
                        "agents": [agent.id, other.id],
                        "resolution": resolution,
                    })

            tick_events["agents"].append(agent.snapshot(self.tick))

        # Record decisions from this tick
        tick_events["decisions"] = [
            asdict(d) for d in self.jev.log if d.tick == self.tick
        ]

        self.timeline.append(tick_events)

    def run(self, ticks: int = SIM_TICKS):
        # Record initial state
        self.timeline.append({
            "tick": 0,
            "agents": [a.snapshot(0) for a in self.agents.values()],
            "decisions": [],
            "events": [],
        })

        for _ in range(ticks):
            self.step()

        return self.export()

    def export(self) -> dict:
        """Export full simulation data for visualization."""
        return {
            "config": {
                "grid_w": GRID_W,
                "grid_h": GRID_H,
                "ticks": self.tick,
                "gate_threshold": CONFIDENCE_GATE,
            },
            "grid": self.grid,
            "tasks": [
                {
                    "id": t.id,
                    "pickup": t.pickup.to_tuple(),
                    "dropoff": t.dropoff.to_tuple(),
                    "weight": t.weight,
                    "status": t.status.value,
                    "carrier_chain": t.carrier_chain,
                }
                for t in self.tasks
            ],
            "agents_init": [a.to_state_dict() for a in self.agents.values()],
            "timeline": self.timeline,
            "jev_log": [asdict(d) for d in self.jev.log],
            "summary": self._summary(),
        }

    def _summary(self) -> dict:
        delivered = sum(1 for t in self.tasks if t.status == TaskStatus.DELIVERED)
        total_decisions = len(self.jev.log)
        auto_decisions = sum(1 for d in self.jev.log if d.gated == "auto")
        escalated = total_decisions - auto_decisions
        handovers = sum(1 for d in self.jev.log if d.decision_type == DecisionType.HANDOVER_GATE.value)
        conflicts = sum(1 for d in self.jev.log if d.decision_type == DecisionType.CONFLICT_RESOLVE.value)

        return {
            "tasks_delivered": delivered,
            "tasks_total": len(self.tasks),
            "total_decisions": total_decisions,
            "auto_accepted": auto_decisions,
            "escalated": escalated,
            "auto_rate": round(auto_decisions / max(total_decisions, 1), 3),
            "handover_attempts": handovers,
            "conflict_resolutions": conflicts,
            "decision_breakdown": {
                dtype.value: sum(1 for d in self.jev.log if d.decision_type == dtype.value)
                for dtype in DecisionType
            },
        }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    world = World()
    data = world.run(SIM_TICKS)

    with open("/home/user/simulation/sim_output.json", "w") as f:
        json.dump(data, f, indent=2)

    s = data["summary"]
    print("═══ Jev Consensus Simulation Complete ═══")
    print(f"  Ticks:              {data['config']['ticks']}")
    print(f"  Tasks delivered:    {s['tasks_delivered']} / {s['tasks_total']}")
    print(f"  Total Jev decisions:{s['total_decisions']}")
    print(f"  Auto-accepted:      {s['auto_accepted']} ({s['auto_rate']*100:.1f}%)")
    print(f"  Escalated:          {s['escalated']}")
    print(f"  Handover attempts:  {s['handover_attempts']}")
    print(f"  Conflicts resolved: {s['conflict_resolutions']}")
    print(f"  Decision breakdown:")
    for dtype, count in s["decision_breakdown"].items():
        print(f"    {dtype:20s}: {count}")
    print(f"\n  Output: /home/user/simulation/sim_output.json")
