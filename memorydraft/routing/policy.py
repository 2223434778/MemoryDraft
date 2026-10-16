from dataclasses import dataclass


METHODS = ("AR", "D", "R", "RD", "H", "RHD", "D_native", "D_adapter", "PLD")


@dataclass
class RouteStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    committed: int = 0
    cost_per_token: float | None = None


class Router:
    def __init__(self, method: str, has_history: bool, probe_rounds: int = 2, cooldown: int = 3):
        if method not in METHODS:
            raise ValueError(f"Unsupported method: {method}")
        self.method, self.has_history = method, has_history
        self.probe_rounds, self.cooldown = probe_rounds, cooldown
        self.stats = {route: RouteStats() for route in ("R", "H", "D", "AR")}
        self.round = 0
        self.r_blocked_until = 0
        self.last_route = None
        self.residence = 0

    def choose(self, has_match: bool) -> str:
        if self.method in {"AR", "D", "H"}:
            return self.method
        if has_match and self.round >= self.r_blocked_until:
            return "R"
        if self.method == "R":
            return "AR"
        if self.method == "RD":
            return "D"
        # Explicitly timed initial probes, then causal EMA costs. No oracle future labels.
        eligible = ("H", "D", "AR") if self.has_history else ("D", "AR")
        for route in eligible:
            if self.stats[route].rounds < self.probe_rounds:
                return route
        if self.last_route in eligible and self.residence < 2:
            return self.last_route
        return min(eligible, key=lambda r: self.stats[r].cost_per_token)

    def update(self, route: str, proposed: int, accepted: int, committed: int, seconds: float):
        stats = self.stats[route]
        stats.rounds += 1
        stats.proposed += proposed
        stats.accepted += accepted
        stats.committed += committed
        cost = seconds / max(committed, 1)
        stats.cost_per_token = cost if stats.cost_per_token is None else 0.7 * stats.cost_per_token + 0.3 * cost
        self.round += 1
        self.residence = self.residence + 1 if self.last_route == route else 1
        self.last_route = route
        if route == "R" and accepted == 0:
            self.r_blocked_until = self.round + self.cooldown
