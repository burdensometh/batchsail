"""Per-device LRU residencies and single active workspace reservations."""

from collections import OrderedDict
from dataclasses import dataclass

from ..errors import Conflict
from ..registry.model import ModelKey
from .maintenance import TrimPlan
from .policy import DeviceSpec
from .types import DeviceLease, DeviceRejection, MemoryDemand, Placement, Residency


@dataclass(frozen=True, slots=True)
class DeviceView:
    name: str
    capacity_bytes: int
    resident_bytes: int
    workspace_bytes: int
    residencies: tuple[Residency, ...]
    lease_token: int | None
    evictions: int
    resets: int

    @property
    def available_bytes(self) -> int:
        return self.capacity_bytes - self.resident_bytes - self.workspace_bytes


class _Device:
    def __init__(self, spec: DeviceSpec):
        self.spec = spec
        self.residents: OrderedDict[ModelKey, Residency] = OrderedDict()
        self.resident_bytes = 0
        self.active: DeviceLease | None = None
        self.epoch = 0
        self.evictions = 0
        self.resets = 0

    def assess(
        self, demand: MemoryDemand, protected: frozenset[ModelKey],
    ) -> Placement | DeviceRejection:
        name = self.spec.name
        capacity = self.spec.capacity_bytes
        minimum = demand.weight_bytes + demand.workspace_bytes
        if minimum > capacity:
            return DeviceRejection(name, "capacity", minimum - capacity)
        if self.active is not None:
            return DeviceRejection(name, "busy", 0)
        resident = self.residents.get(demand.key)
        if resident is not None and resident.weight_bytes != demand.weight_bytes:
            raise RuntimeError("weight footprint changed for a resident immutable revision")

        required = self.resident_bytes + demand.workspace_bytes
        if resident is None:
            required += demand.weight_bytes
        victims: list[Residency] = []
        blocked: list[ModelKey] = []
        for entry in self.residents.values():
            if required <= capacity:
                break
            if entry.key == demand.key:
                continue
            if entry.key in protected:
                blocked.append(entry.key)
                continue
            required -= entry.weight_bytes
            victims.append(entry)
        if required > capacity:
            return DeviceRejection(name, "protected_residency", required - capacity, tuple(blocked))
        return Placement(name, self.epoch, demand, tuple(victims), resident is not None, required)

    def acquire(self, placement: Placement, token: int) -> DeviceLease:
        if placement.epoch != self.epoch or self.active is not None:
            raise Conflict(f"device placement for {self.spec.name} is stale")
        for victim in placement.victims:
            self.resident_bytes -= self.residents.pop(victim.key).weight_bytes
            self.evictions += 1
        demand = placement.demand
        if placement.cached:
            residency = self.residents[demand.key]
            self.residents.move_to_end(demand.key)
        else:
            residency = Residency(demand.key, token, demand.weight_bytes)
            self.residents[demand.key] = residency
            self.resident_bytes += residency.weight_bytes
        self.active = DeviceLease(
            self.spec.name, token, residency, demand.workspace_bytes, tuple(self.residents.values()),
        )
        self.epoch += 1
        self._check_invariant()
        return self.active

    def release(self, lease: DeviceLease, *, reset: bool) -> None:
        if self.active is not lease:
            raise RuntimeError(f"lease {lease.token} is no longer active on {lease.device}")
        self.active = None
        if reset:
            self.reset_cache()
            return
        self.epoch += 1
        self._check_invariant()

    def plan_trim(
        self, protected: frozenset[ModelKey], *, target: ModelKey | None,
    ) -> TrimPlan | None:
        if self.active is not None:
            return None
        victims: list[Residency] = []
        keep: list[Residency] = []
        for entry in self.residents.values():
            if entry.key not in protected and (target is None or target == entry.key):
                victims.append(entry)
            else:
                keep.append(entry)
        return TrimPlan(self.spec.name, self.epoch, tuple(victims), tuple(keep))

    def apply_trim(self, plan: TrimPlan, retained: tuple[Residency, ...]) -> tuple[Residency, ...]:
        if self.epoch != plan.epoch or self.active is not None:
            raise Conflict(f"cache trim for {self.spec.name} is stale")
        kept = set(retained)
        if not kept.issubset(plan.keep):
            raise RuntimeError(f"device {self.spec.name} acknowledged unexpected residencies")
        # Drop accounting for allowed but absent kernels, too. The device could
        # have reserved space before a kernel load failed.
        removed = tuple(entry for entry in self.residents.values() if entry not in kept)
        for entry in removed:
            self.resident_bytes -= self.residents.pop(entry.key).weight_bytes
        self.evictions += len(removed)
        self.epoch += 1
        self._check_invariant()
        return removed

    def reset_cache(self) -> tuple[Residency, ...]:
        if self.active is not None:
            raise Conflict(f"cannot reset ledger cache on leased device {self.spec.name}")
        removed = tuple(self.residents.values())
        self.residents.clear()
        self.resident_bytes = 0
        self.resets += 1
        self.epoch += 1
        self._check_invariant()
        return removed

    def _check_invariant(self) -> None:
        workspace = 0 if self.active is None else self.active.workspace_bytes
        if not 0 <= self.resident_bytes <= self.spec.capacity_bytes - workspace:
            raise RuntimeError("device memory ledger violated its capacity")

    def view(self) -> DeviceView:
        lease = self.active
        return DeviceView(
            name=self.spec.name,
            capacity_bytes=self.spec.capacity_bytes,
            resident_bytes=self.resident_bytes,
            workspace_bytes=0 if lease is None else lease.workspace_bytes,
            residencies=tuple(self.residents.values()),
            lease_token=None if lease is None else lease.token,
            evictions=self.evictions,
            resets=self.resets,
        )


class DeviceLedger:
    """The coordinator serializes plan, acquire and release operations."""

    def __init__(self, specs: tuple[DeviceSpec, ...]):
        self._devices = {spec.name: _Device(spec) for spec in specs}
        if not self._devices or len(self._devices) != len(specs):
            raise ValueError("a device ledger needs one or more uniquely named devices")
        self._token = 0
        self.max_capacity = max(spec.capacity_bytes for spec in specs)

    def assess(
        self, demand: MemoryDemand, protected: frozenset[ModelKey],
    ) -> tuple[Placement | DeviceRejection, ...]:
        return tuple(device.assess(demand, protected) for device in self._devices.values())

    @staticmethod
    def preferred(placements: tuple[Placement | DeviceRejection, ...]) -> Placement | None:
        return min(
            (candidate for candidate in placements if isinstance(candidate, Placement)),
            key=lambda plan: (not plan.cached, plan.eviction_bytes, plan.bytes_after, plan.device),
            default=None,
        )

    def acquire(self, placement: Placement, protected: frozenset[ModelKey]) -> DeviceLease:
        device = self._devices[placement.device]
        if device.assess(placement.demand, protected) != placement:
            raise Conflict(f"device placement for {placement.device} is stale or ineligible")
        self._token += 1
        return device.acquire(placement, self._token)

    def release(self, lease: DeviceLease, *, reset: bool = False) -> None:
        self._devices[lease.device].release(lease, reset=reset)

    def has_idle_device(self) -> bool:
        return any(device.active is None for device in self._devices.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._devices)

    def plan_trim(
        self, device: str, protected: frozenset[ModelKey], *, target: ModelKey | None = None,
    ) -> TrimPlan | None:
        return self._devices[device].plan_trim(protected, target=target)

    def apply_trim(self, plan: TrimPlan, retained: tuple[Residency, ...]) -> tuple[Residency, ...]:
        return self._devices[plan.device].apply_trim(plan, retained)

    def reset_idle(self, device: str) -> tuple[Residency, ...]:
        return self._devices[device].reset_cache()

    def snapshot(self) -> tuple[DeviceView, ...]:
        return tuple(device.view() for device in self._devices.values())
