import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class Params:
    @property
    def dirpath(self) -> Path:
        return Path(
            os.path.join(*[f"{field}-{val}" for field, val in asdict(self).items()])
        )


@dataclass(frozen=True, kw_only=True)
class LUCJParams(Params):
    connectivity: str  # options: all-to-all, linear, square, hex, heavy-hex
    n_reps: int | None
    with_final_orbital_rotation: bool


@dataclass(frozen=True, kw_only=True)
class UCCSDParams(Params):
    with_final_orbital_rotation: bool


@dataclass(frozen=True, kw_only=True)
class LUCJAnglesParams(Params):
    connectivity: str  # options: all-to-all, square, hex, heavy-hex
    n_reps: int | None
    with_final_orbital_rotation: bool
    n_givens_layers: int


@dataclass(frozen=True, kw_only=True)
class LBFGSBParams(Params):
    maxiter: int
    maxfun: int
    maxcor: int = 10
    maxls: int = 20
    eps: float = 1e-8
    ftol: float = 1e-8
    gtol: float = 1e-5


@dataclass(frozen=True, kw_only=True)
class LinearMethodParams(Params):
    maxiter: int
    lindep: float
    epsilon: float
    ftol: float
    gtol: float
    regularization: float
    variation: float
    optimize_regularization: bool
    optimize_variation: bool


@dataclass(frozen=True, kw_only=True)
class StochasticReconfigurationParams(Params):
    maxiter: int
    cond: float
    epsilon: float
    gtol: float
    regularization: float
    variation: float
    optimize_regularization: bool
    optimize_variation: bool
