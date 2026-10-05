import random 
from abc import ABC, abstractmethod
import math
import random
import numpy as np
from nas_framework.population import Population, ABCPopulation, FoodSource, Individual, ABCPopulation, FoodSource
from nas_framework.selection import Selection,TournamentSelection, NeighborSelection, GuidanceSelection
from nas_framework.variation import Variation, MutationOnlyVariation, CrossoverMutationVariation
from nas_framework.crossover import Crossover
from nas_framework.mutation import Mutation
from nas_framework.replacement import Replacement
from nas_framework.evaluator import Evaluator
from nas_framework.termination import Termination, MaxEvaluationsTermination
from nas_framework.history import History
from nas_framework.search_space import SearchSpace
from nas_framework.mo_utils import pareto_front as compute_pareto_front
from nas_framework.mo_utils import exact_pareto_front_2d
from nas_framework.mo_utils import dominates
from nas_framework.mo_utils import assign_rank_and_crowding


class SearchStrategy(ABC):
    """Abstract search strategy that orchestrates population-based components."""

    def __init__(self, population: Population, selection: Selection,
                 variation: Variation, replacement: Replacement,
                 evaluator: Evaluator, termination: Termination | None = None,
                 history: History | None = None, budget: int = 500):
        self.population = population
        self.selection = selection
        self.variation = variation
        self.replacement = replacement
        self.evaluator = evaluator
        self.termination = termination or MaxEvaluationsTermination(budget)
        self.history = history or History()
        self.evaluations: int = 0
        self.generations: int = 0

    @abstractmethod
    def run(self) -> Population:
        """Execute the search and return the final population."""
        ...

    def _record_history(self) -> None:
        self.history.record(
            generation=self.generations,
            evaluations=self.evaluations,
            population=self.population.individuals,
            pareto_front=self.population.pareto_front(),
        )

    def _evaluate_offspring(self, offspring: list[Individual]) -> None:
        for child in offspring:
            child.fitness = self.evaluator.evaluate(child.genotype)
            metadata = {}
            if hasattr(self.population.search_space, "metadata_from_genotype"):
                metadata = self.population.search_space.metadata_from_genotype(child.genotype)
            elif hasattr(self.evaluator.benchmark, "get_metadata"):
                metadata = self.evaluator.benchmark.get_metadata(child.genotype)
            child.metadata = metadata
            self.evaluations += 1
            if self.termination.should_stop(self.evaluations, self.generations):
                break


class GeneticAlgorithm(SearchStrategy):
    """Pareto-based GA: selection â†’ variation â†’ replacement loop."""

    def run(self) -> Population:
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            # Selection
            parents = self.selection.select(
                self.population.individuals,
                self.population.size,
                self.evaluator.objective_directions,
            )

            # Variation
            offspring = self.variation.generate(parents, self.population.size)
            self._evaluate_offspring(offspring)

            # Replacement
            self.population.individuals = self.replacement.replace(
                self.population.individuals,
                offspring,
                self.population.size,
                self.evaluator.objective_directions,
            )
            self.generations += 1
            self._record_history()

        return self.population


class EvolutionStrategy(SearchStrategy):
    """(mu, lambda) ES: mutation-only variation."""

    def __init__(self, population: Population, selection: Selection,
                 crossover: Crossover, mutation: Mutation,
                 replacement: Replacement, evaluator: Evaluator,
                 budget: int = 500, n_offspring: int = 40,
                 termination: Termination | None = None,
                 history: History | None = None):
        variation = MutationOnlyVariation(mutation)
        super().__init__(population, selection, variation,
                         replacement, evaluator,
                         termination=termination, history=history,
                         budget=budget)
        self.n_offspring = n_offspring

    def run(self) -> Population:
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            parents = self.selection.select(
                self.population.individuals,
                self.population.size,
                self.evaluator.objective_directions,
            )

            offspring = self.variation.generate(parents, self.n_offspring)
            self._evaluate_offspring(offspring)

            self.population.individuals = self.replacement.replace(
                self.population.individuals,
                offspring,
                self.population.size,
                self.evaluator.objective_directions,
            )
            self.generations += 1
            self._record_history()

        return self.population


class RandomSearch(SearchStrategy):
    """Random search baseline  samples new architectures each iteration."""

    def __init__(self, population: Population, selection: Selection,
                 crossover: Crossover, mutation: Mutation,
                 replacement: Replacement, evaluator: Evaluator,
                 budget: int = 500, termination: Termination | None = None,
                 history: History | None = None):
        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(population, selection, variation, replacement,
                         evaluator, termination=termination,
                         history=history, budget=budget)

    def run(self) -> Population:
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            geno = self.population.search_space.random_individual()
            fit = self.evaluator.evaluate(geno)
            self.evaluations += 1
            metadata = {}
            if hasattr(self.population.search_space, "metadata_from_genotype"):
                metadata = self.population.search_space.metadata_from_genotype(geno)
            elif hasattr(self.evaluator.benchmark, "get_metadata"):
                metadata = self.evaluator.benchmark.get_metadata(geno)
            child = Individual(geno, fit, metadata=metadata)

            self.population.individuals = self.replacement.replace(
                self.population.individuals,
                [child],
                self.population.size,
                self.evaluator.objective_directions,
            )
            self.generations += 1
            self._record_history()

        return self.population


class BruteForceParetoSearch:
    """Exhaustive strategy that returns the optimal Pareto front on finite spaces."""

    def __init__(self, search_space: SearchSpace, evaluator: Evaluator,
                 history: History | None = None):
        self.search_space = search_space
        self.evaluator = evaluator
        self.history = history or History()
        self.evaluations: int = 0
        self.generations: int = 0
        self.population: list[Individual] = []

    def run(self) -> list[Individual]:
        if not hasattr(self.search_space, "all_genotypes"):
            raise TypeError(
                "BruteForceParetoSearch requires a finite search space exposing "
                "all_genotypes()."
            )

        individuals: list[Individual] = []
        for genotype in self.search_space.all_genotypes():
            fitness = self.evaluator.evaluate(genotype)
            metadata = {}
            if hasattr(self.search_space, "metadata_from_genotype"):
                metadata = self.search_space.metadata_from_genotype(genotype)
            elif hasattr(self.evaluator.benchmark, "get_metadata"):
                metadata = self.evaluator.benchmark.get_metadata(genotype)
            individuals.append(Individual(genotype[:], fitness, metadata=metadata))

        self.population = individuals
        self.evaluations = len(individuals)
        self.generations = 1

        front = compute_pareto_front(individuals, self.evaluator.objective_directions)
        self.history.record(
            generation=0,
            evaluations=self.evaluations,
            population=individuals,
            pareto_front=front,
        )
        return front


class SkylineSearch:
    """Exact Pareto strategy for 2 objectives using sort+sweep skyline."""

    def __init__(self, search_space: SearchSpace, evaluator: Evaluator,
                 history: History | None = None):
        self.search_space = search_space
        self.evaluator = evaluator
        self.history = history or History()
        self.evaluations: int = 0
        self.generations: int = 0
        self.population: list[Individual] = []

    def run(self) -> list[Individual]:
        if not hasattr(self.search_space, "all_genotypes"):
            raise TypeError(
                "SkylineSearch requires a finite search space exposing "
                "all_genotypes()."
            )
        if len(self.evaluator.objective_directions) != 2:
            raise ValueError("SkylineSearch supports exactly 2 objectives.")

        individuals: list[Individual] = []
        for genotype in self.search_space.all_genotypes():
            fitness = self.evaluator.evaluate(genotype)
            metadata = {}
            if hasattr(self.search_space, "metadata_from_genotype"):
                metadata = self.search_space.metadata_from_genotype(genotype)
            elif hasattr(self.evaluator.benchmark, "get_metadata"):
                metadata = self.evaluator.benchmark.get_metadata(genotype)
            individuals.append(Individual(genotype[:], fitness, metadata=metadata))

        self.population = individuals
        self.evaluations = len(individuals)
        self.generations = 1

        directions_2d = (
            self.evaluator.objective_directions[0],
            self.evaluator.objective_directions[1],
        )
        front = exact_pareto_front_2d(individuals, directions_2d)
        self.history.record(
            generation=0,
            evaluations=self.evaluations,
            population=individuals,
            pareto_front=front,
        )
        return front
 
# Implementation of the method described in the article Multi-Objective
#  White Shark Optimizer for Global Optimization and Rural Sports-Facilities
#  Location Problem
class MOWSOSearch:
    """Multi-Objective White Shark Optimizer with archive true-distance pruning.

    This follows the paper workflow with:
    - White-shark speed and movement update phases
    - External non-dominated archive
    - True-distance based archive truncation
    - Guide selection as archive member with maximum true distance
    """

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Evaluator,
        pop_size: int = 50,
        max_iterations: int = 300,
        archive_size: int | None = None,
        history: History | None = None,
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.archive_size = archive_size or pop_size
        self.history = history or History()

        self.evaluations: int = 0
        self.generations: int = 0
        self.population: list[Individual] = []
        self.archive: list[Individual] = []

        # Parameters reported in the referenced WSO/MOWSO setup.
        self.p_min = 0.5
        self.p_max = 1.5
        self.tau = 4.125
        self.f_min = 0.07
        self.f_max = 0.75
        self.a0 = 6.25
        self.a1 = 100.0
        self.a2 = 0.0005

    def _mu(self) -> float:
        # Eq. (5): contraction factor.
        inner = self.tau * self.tau - 4.0 * self.tau
        inner = max(inner, 0.0)
        denom = abs(2.0 - self.tau - math.sqrt(inner))
        if denom <= 1e-12:
            return 1.0
        return 2.0 / denom

    def _p1_p2(self, k: int, K: int) -> tuple[float, float]:
        # Eq. (3) and Eq. (4) schedule used in the paper.
        expo = math.exp(-((4.0 * k) / K) ** 2)
        p1 = self.p_max + (self.p_max - self.p_min) * expo
        p2 = self.p_min + (self.p_max - self.p_min) * expo
        return p1, p2

    def _mv(self, k: int, K: int) -> float:
        # Eq. (11): movement force.
        return 1.0 / (self.a0 + math.exp((K / 2.0 - k) / self.a1))

    def _f(self) -> float:
        # Eq. (10) form used in the paper text.
        return self.f_min + (self.f_max - self.f_min) / (self.f_max + self.f_min)

    def _ss(self, k: int, K: int) -> float:
        # Eq. (14): olfactory/visual intensity.
        return abs(1.0 - math.exp(-self.a2 * k / K))

    def _bounds(self) -> tuple[float, float]:
        return 0.0, float(self.search_space.num_ops - 1)

    def _clip_position(self, pos: list[float]) -> list[float]:
        lower, upper = self._bounds()
        return [min(upper, max(lower, x)) for x in pos]

    def _to_genotype(self, pos: list[float]) -> list[int]:
        lower, upper = self._bounds()
        return [int(min(upper, max(lower, round(x)))) for x in pos]

    def _individual_from_pos(self, pos: list[float]) -> Individual:
        genotype = self._to_genotype(pos)
        fitness = self.evaluator.evaluate(genotype)
        metadata = {}
        if hasattr(self.search_space, "metadata_from_genotype"):
            metadata = self.search_space.metadata_from_genotype(genotype)
        elif hasattr(self.evaluator.benchmark, "get_metadata"):
            metadata = self.evaluator.benchmark.get_metadata(genotype)
        self.evaluations += 1
        return Individual(genotype, fitness, metadata=metadata)

    def _dominates(self, a: Individual, b: Individual) -> bool:
        return dominates(a, b, self.evaluator.objective_directions)

    def _distance(self, a: Individual, b: Individual) -> float:
        if a.fitness is None or b.fitness is None:
            return float("inf")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a.fitness, b.fitness)))

    def _true_distance_scores(self, inds: list[Individual]) -> list[float]:
        if len(inds) <= 1:
            return [float("inf")] * len(inds)
        scores: list[float] = []
        for i, ind_i in enumerate(inds):
            dists = [self._distance(ind_i, inds[j]) for j in range(len(inds)) if j != i]
            dists.sort()
            scores.append(sum(dists))
        return scores

    def _archive_update(self, candidates: list[Individual]) -> None:
        # Merge by genotype key to avoid duplicates.
        merged_by_key: dict[tuple[int, ...], Individual] = {}
        for ind in self.archive + candidates:
            merged_by_key[tuple(ind.genotype)] = ind
        merged = list(merged_by_key.values())

        nd = compute_pareto_front(merged, self.evaluator.objective_directions)
        self.archive = nd

        while len(self.archive) > self.archive_size:
            td = self._true_distance_scores(self.archive)
            remove_idx = min(range(len(self.archive)), key=lambda i: td[i])
            del self.archive[remove_idx]

    def _guide_position(self, positions: list[list[float]]) -> list[float]:
        if not self.archive:
            return random.choice(positions)[:]
        td = self._true_distance_scores(self.archive)
        guide = self.archive[max(range(len(self.archive)), key=lambda i: td[i])]
        return [float(x) for x in guide.genotype]

    def run(self) -> list[Individual]:
        K = max(1, int(self.max_iterations))
        mu = self._mu()

        # Initialization
        positions: list[list[float]] = []
        velocities: list[list[float]] = []
        individuals: list[Individual] = []
        for _ in range(self.pop_size):
            geno = self.search_space.random_individual()
            pos = [float(x) for x in geno]
            ind = self._individual_from_pos(pos)
            positions.append(pos)
            velocities.append([0.0] * len(pos))
            individuals.append(ind)

        self.population = individuals
        self.archive = []
        self._archive_update(individuals)

        pbest_pos = [p[:] for p in positions]
        pbest_ind = [Individual(ind.genotype[:], ind.fitness, ind.metadata.copy()) for ind in individuals]

        self.generations = 0
        self.history.record(
            generation=self.generations,
            evaluations=self.evaluations,
            population=individuals,
            pareto_front=self.archive,
        )

        for k in range(1, K + 1):
            p1, p2 = self._p1_p2(k, K)
            mv = self._mv(k, K)
            f = self._f()
            ss = self._ss(k, K)
            guide = self._guide_position(positions)

            next_positions: list[list[float]] = []
            next_individuals: list[Individual] = []

            for i in range(self.pop_size):
                v_idx = random.randint(0, self.pop_size - 1)
                c1 = random.random()
                c2 = random.random()

                cur = positions[i]
                vel = velocities[i]
                pbest_v = pbest_pos[v_idx]

                new_vel = []
                for d in range(len(cur)):
                    step = vel[d] + p1 * (guide[d] - cur[d]) * c1 + p2 * (pbest_v[d] - cur[d]) * c2
                    new_vel.append(mu * step)
                velocities[i] = new_vel

                if random.random() < mv:
                    # Eq. (6), (7), (8), (9): boundary-aware movement.
                    lower, upper = self._bounds()
                    new_pos = []
                    for d in range(len(cur)):
                        a = 1.0 if cur[d] > upper else 0.0
                        b = 1.0 if cur[d] < lower else 0.0
                        w_o = 1.0 if (bool(a) ^ bool(b)) else 0.0
                        new_pos.append(cur[d] * (1.0 - w_o) + upper * a + lower * b)
                else:
                    new_pos = [cur[d] + new_vel[d] / f for d in range(len(cur))]

                if random.random() < ss:
                    r1 = random.random()
                    r2 = random.random()
                    rand_dw = random.random()
                    sign_dir = 1.0 if r2 >= 0.5 else -1.0
                    d_w = [abs(rand_dw * (guide[d] - cur[d])) for d in range(len(cur))]
                    w_prime = [guide[d] + r1 * d_w[d] * sign_dir for d in range(len(cur))]
                    denom = 2.0 * max(random.random(), 1e-12)
                    new_pos = [(new_pos[d] + w_prime[d]) / denom for d in range(len(cur))]

                new_pos = self._clip_position(new_pos)
                ind_new = self._individual_from_pos(new_pos)

                if self._dominates(ind_new, pbest_ind[i]):
                    pbest_pos[i] = new_pos[:]
                    pbest_ind[i] = Individual(ind_new.genotype[:], ind_new.fitness, ind_new.metadata.copy())
                elif (not self._dominates(pbest_ind[i], ind_new)) and random.random() < 0.5:
                    pbest_pos[i] = new_pos[:]
                    pbest_ind[i] = Individual(ind_new.genotype[:], ind_new.fitness, ind_new.metadata.copy())

                next_positions.append(new_pos)
                next_individuals.append(ind_new)

            positions = next_positions
            individuals = next_individuals
            self.population = individuals
            self._archive_update(individuals)

            self.generations = k
            self.history.record(
                generation=self.generations,
                evaluations=self.evaluations,
                population=individuals,
                pareto_front=self.archive,
            )

        return [Individual(ind.genotype[:], ind.fitness, ind.metadata.copy()) for ind in self.archive]


class MOSHOSearch:
    """Multi-Objective Shark Hunting Optimization for NAS-Bench-201."""

    OPS = [
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]
    GENE_SIZE = 6
    GENE_VALUES = list(range(len(OPS)))
    SUPPORTED_UNITS = frozenset({"U01", "U02", "U03", "U04", "U05", "U06", "U07"})

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Evaluator,
        pop_size: int = 50,
        max_iterations: int = 300,
        archive_size: int | None = None,
        e0: float = 0.818,
        e_min: float = 0.508,
        e_max: float = 1.699,
        eta: float = 15.0,
        delta: float = 0.0968,
        disabled_units: set[str] | None = None,
        history: History | None = None,
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.archive_size = archive_size or pop_size
        self.history = history or History()

        self.e0 = e0
        self.e_min = e_min
        self.e_max = e_max
        self.eta = eta
        self.delta = delta
        self.disabled_units = frozenset(disabled_units or ())

        unknown_units = self.disabled_units - self.SUPPORTED_UNITS
        if unknown_units:
            raise ValueError(f"Unsupported MOSHO ablation units: {sorted(unknown_units)}")

        if self.search_space.num_edges != self.GENE_SIZE:
            raise ValueError(
                f"MOSHOSearch expects {self.GENE_SIZE} genes, got "
                f"{self.search_space.num_edges}."
            )
        if self.search_space.num_ops != len(self.GENE_VALUES):
            raise ValueError(
                f"MOSHOSearch expects {len(self.GENE_VALUES)} operations, got "
                f"{self.search_space.num_ops}."
            )

        self.base_probs = {
            "patrol": 0.20,
            "scent": 0.25,
            "circle": 0.15,
            "burst": 0.10,
            "crossover": 0.20,
            "scout": 0.10,
        }
        self._op_credit: dict[str, float] = {op: 0.5 for op in self.base_probs}

        self.evaluations: int = 0
        self.generations: int = 0
        self.population: list[Individual] = []
        self.archive: list[Individual] = []
        self._energy: list[float] = []

        self._archive_sampler_dirty = True
        self._archive_sampler_weights: list[float] = []

    def _clone_individual(self, ind: Individual) -> Individual:
        metadata = ind.metadata.copy() if ind.metadata else {}
        return Individual(ind.genotype[:], ind.fitness, metadata=metadata)

    def _evaluate_arch(self, arch: list[int]) -> Individual:
        fitness = self.evaluator.evaluate(arch)
        metadata = {}
        if hasattr(self.search_space, "metadata_from_genotype"):
            metadata = self.search_space.metadata_from_genotype(arch)
        elif hasattr(self.evaluator.benchmark, "get_metadata"):
            metadata = self.evaluator.benchmark.get_metadata(arch)
        self.evaluations += 1
        return Individual(arch[:], fitness, metadata=metadata)

    def _random_arch(self) -> list[int]:
        return self.search_space.random_individual()

    def _mutate_gene(self, arch: list[int], idx: int) -> list[int]:
        new_arch = arch[:]
        choices = [v for v in self.GENE_VALUES if v != new_arch[idx]]
        new_arch[idx] = random.choice(choices)
        return new_arch

    def _unit_enabled(self, code: str) -> bool:
        return code not in self.disabled_units

    def _continuous_quantize(self, values: list[float]) -> list[int]:
        lower = float(min(self.GENE_VALUES))
        upper = float(max(self.GENE_VALUES))
        return [
            int(min(upper, max(lower, round(v))))
            for v in values
        ]

    def _continuous_like_candidate(
        self,
        arch: list[int],
        *,
        target: list[int] | None = None,
        partner: list[int] | None = None,
        noise_scale: float = 1.0,
    ) -> list[int]:
        relaxed = [float(g) for g in arch]
        anchor = target or self._random_arch()
        mate = partner or random.choice(self.population).genotype

        for j in range(self.GENE_SIZE):
            step = random.uniform(-noise_scale, noise_scale)
            step += 0.55 * (anchor[j] - relaxed[j])
            step += 0.25 * (mate[j] - relaxed[j])
            relaxed[j] += step
        return self._continuous_quantize(relaxed)

    def _crowding_scores(self, inds: list[Individual]) -> list[float]:
        if not inds:
            return []
        if len(inds) <= 2:
            return [float("inf")] * len(inds)

        scores = [0.0 for _ in inds]
        n_obj = len(self.evaluator.objective_directions)
        for obj in range(n_obj):
            direction = self.evaluator.objective_directions[obj]
            order = sorted(
                range(len(inds)),
                key=lambda i: inds[i].fitness[obj] * direction,
            )
            scores[order[0]] = float("inf")
            scores[order[-1]] = float("inf")

            min_v = inds[order[0]].fitness[obj] * direction
            max_v = inds[order[-1]].fitness[obj] * direction
            if max_v == min_v:
                continue

            for k in range(1, len(order) - 1):
                idx = order[k]
                if scores[idx] == float("inf"):
                    continue
                prev_v = inds[order[k - 1]].fitness[obj] * direction
                next_v = inds[order[k + 1]].fitness[obj] * direction
                scores[idx] += (next_v - prev_v) / (max_v - min_v)
        return scores

    def _true_distance_scores(self, inds: list[Individual]) -> list[float]:
        if len(inds) <= 1:
            return [float("inf")] * len(inds)

        scores: list[float] = []
        for i, ind_i in enumerate(inds):
            if ind_i.fitness is None:
                scores.append(float("inf"))
                continue

            total = 0.0
            for j, ind_j in enumerate(inds):
                if i == j or ind_j.fitness is None:
                    continue
                total += math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(ind_i.fitness, ind_j.fitness))
                )
            scores.append(total)
        return scores

    def _archive_update(self, candidate: Individual) -> bool:
        for archived in self.archive:
            if dominates(archived, candidate, self.evaluator.objective_directions):
                return False

        kept: list[Individual] = []
        changed = False
        for archived in self.archive:
            if dominates(candidate, archived, self.evaluator.objective_directions):
                changed = True
                continue
            kept.append(archived)

        if any(
            tuple(archived.genotype) == tuple(candidate.genotype)
            and archived.fitness == candidate.fitness
            for archived in kept
        ):
            return changed

        kept.append(self._clone_individual(candidate))
        self.archive = kept
        self._archive_sampler_dirty = True
        return True

    def _truncate_archive(self) -> None:
        if len(self.archive) <= self.archive_size:
            return

        if not self._unit_enabled("U03"):
            while len(self.archive) > self.archive_size:
                scores = self._true_distance_scores(self.archive)
                remove_idx = min(range(len(self.archive)), key=lambda i: scores[i])
                del self.archive[remove_idx]
            self._archive_sampler_dirty = True
            return

        scores = self._crowding_scores(self.archive)
        order = sorted(range(len(self.archive)), key=lambda i: scores[i], reverse=True)
        keep = set(order[: self.archive_size])
        self.archive = [self.archive[i] for i in range(len(self.archive)) if i in keep]
        self._archive_sampler_dirty = True

    def _refresh_archive_sampler(self) -> None:
        if not self.archive:
            self._archive_sampler_weights = []
            self._archive_sampler_dirty = False
            return

        scores = self._crowding_scores(self.archive)
        self._archive_sampler_weights = [
            10.0 if math.isinf(score) else max(1e-3, score) for score in scores
        ]
        self._archive_sampler_dirty = False

    def _sample_archive(self) -> Individual:
        if self._archive_sampler_dirty or len(self._archive_sampler_weights) != len(self.archive):
            self._refresh_archive_sampler()
        if not self.archive:
            raise ValueError("Cannot sample from an empty archive.")

        if not self._unit_enabled("U07"):
            scores = self._true_distance_scores(self.archive)
            leader_idx = max(range(len(self.archive)), key=lambda i: scores[i])
            return self.archive[leader_idx]
        return random.choices(self.archive, weights=self._archive_sampler_weights, k=1)[0]

    def _patrol(self, arch: list[int]) -> list[int]:
        if not self._unit_enabled("U01"):
            return self._continuous_like_candidate(arch, noise_scale=1.35)

        u = max(1e-9, random.random())
        k = int(min(self.GENE_SIZE, max(2, round((u ** -0.35) % self.GENE_SIZE))))
        k = min(self.GENE_SIZE, max(2, k))
        idxs = random.sample(range(self.GENE_SIZE), k=k)
        new_arch = arch[:]
        for idx in idxs:
            new_arch = self._mutate_gene(new_arch, idx)
        return new_arch

    def _scent(self, arch: list[int]) -> list[int]:
        target = self._sample_archive().genotype if self.archive else self._random_arch()
        partner = random.choice(self.population).genotype

        if not self._unit_enabled("U01"):
            return self._continuous_like_candidate(
                arch,
                target=target,
                partner=partner,
                noise_scale=0.8,
            )

        new_arch = arch[:]
        for j in range(self.GENE_SIZE):
            r = random.random()
            if r < 0.55:
                new_arch[j] = target[j]
            elif r < 0.75:
                new_arch[j] = partner[j]
            elif r < 0.80:
                new_arch[j] = random.choice(self.GENE_VALUES)
        return new_arch

    def _circle(self, arch: list[int]) -> list[int]:
        if not self._unit_enabled("U01"):
            return self._continuous_like_candidate(arch, noise_scale=0.55)

        k = 1 if random.random() < 0.7 else 2
        idxs = random.sample(range(self.GENE_SIZE), k=k)
        new_arch = arch[:]
        for idx in idxs:
            new_arch = self._mutate_gene(new_arch, idx)
        return new_arch

    def _burst(self, arch: list[int]) -> list[int]:
        target = self._sample_archive().genotype if self.archive else self._random_arch()

        if not self._unit_enabled("U01"):
            return self._continuous_like_candidate(
                arch,
                target=target,
                noise_scale=1.05,
            )

        k = max(1, int(round(0.33 * self.GENE_SIZE)))
        copy_idx = set(random.sample(range(self.GENE_SIZE), k=k))

        new_arch = [0] * self.GENE_SIZE
        for j in range(self.GENE_SIZE):
            if j in copy_idx:
                new_arch[j] = target[j]
            else:
                new_arch[j] = arch[j] if random.random() < 0.65 else random.choice(self.GENE_VALUES)
        return new_arch

    def _crossover(self, arch: list[int]) -> list[int]:
        if not self._unit_enabled("U02"):
            return arch[:]

        if self.archive and random.random() < 0.55:
            partner = self._sample_archive().genotype
        else:
            partner = random.choice(self.population).genotype

        child = [arch[j] if random.random() < 0.5 else partner[j] for j in range(self.GENE_SIZE)]
        if random.random() < 0.3:
            idx = random.randrange(self.GENE_SIZE)
            child[idx] = random.choice([v for v in self.GENE_VALUES if v != child[idx]])
        return child

    def _scout(self) -> list[int]:
        if not self._unit_enabled("U01"):
            return self._continuous_like_candidate(self._random_arch(), noise_scale=1.5)
        return self._random_arch()

    def _choose_operator(self, probs: dict[str, float]) -> str:
        ops = list(probs.keys())
        weights = list(probs.values())
        return random.choices(ops, weights=weights, k=1)[0]

    def _normalize_probs(self, probs: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, value) for value in probs.values())
        if total <= 0.0:
            n = max(1, len(probs))
            return {key: 1.0 / n for key in probs}
        return {key: max(0.0, value) / total for key, value in probs.items()}

    def _operator_probability_adaptation(
        self,
        probs: dict[str, float],
        imp_rate: float,
        progress: float,
    ) -> dict[str, float]:
        p = max(0.0, min(1.0, progress))
        improved = max(0.0, min(1.0, imp_rate))

        adjusted = dict(probs)
        if "scout" in adjusted:
            adjusted["scout"] *= 1.0 + (1.0 - p) * (1.0 - improved)
        if "patrol" in adjusted:
            adjusted["patrol"] *= 1.0 + 0.6 * (1.0 - p)
        if "scent" in adjusted:
            adjusted["scent"] *= 1.0 + 0.6 * p
        if "crossover" in adjusted:
            adjusted["crossover"] *= 1.0 + 0.5 * p
        if "burst" in adjusted:
            adjusted["burst"] *= 1.0 + 0.3 * max(0.0, 0.2 - improved)
        if "circle" in adjusted:
            adjusted["circle"] *= 1.0 + 0.25 * improved

        return self._normalize_probs(adjusted)

    def _fixed_probability_schedule(self, progress: float) -> dict[str, float]:
        p = max(0.0, min(1.0, progress))
        adjusted = dict(self.base_probs)
        if "patrol" in adjusted:
            adjusted["patrol"] *= 1.0 + 0.5 * (1.0 - p)
        if "scent" in adjusted:
            adjusted["scent"] *= 1.0 + 0.4 * p
        if "circle" in adjusted:
            adjusted["circle"] *= 1.0 + 0.1 * (1.0 - p)
        if "burst" in adjusted:
            adjusted["burst"] *= 1.0 + 0.2 * p
        if "scout" in adjusted:
            adjusted["scout"] *= 1.0 + 0.25 * (1.0 - p)
        return self._normalize_probs(adjusted)

    def _credit_biased_probs(self, probs: dict[str, float], progress: float) -> dict[str, float]:
        if not self._unit_enabled("U07"):
            return probs
        if not probs:
            return probs

        p = max(0.0, min(1.0, progress))
        mean_credit = sum(self._op_credit.get(op, 0.5) for op in probs) / max(1, len(probs))
        strength = 0.35 + 0.35 * p

        adjusted: dict[str, float] = {}
        for op, base_p in probs.items():
            credit = self._op_credit.get(op, 0.5)
            multiplier = 1.0 + strength * (credit - mean_credit)
            adjusted[op] = max(1e-9, base_p * multiplier)

        adjusted = self._normalize_probs(adjusted)

        floor = 0.04
        n = len(adjusted)
        if floor * n < 1.0:
            rem = 1.0 - floor * n
            adjusted = {op: floor + rem * prob for op, prob in adjusted.items()}
            adjusted = self._normalize_probs(adjusted)

        return adjusted

    def _update_operator_credit(
        self,
        op: str,
        accepted: bool,
        archive_improved: bool,
        dominates_current: bool,
    ) -> None:
        if not self._unit_enabled("U07"):
            return
        reward = 0.0
        if archive_improved:
            reward = 1.0
        elif dominates_current and accepted:
            reward = 0.8
        elif accepted:
            reward = 0.35

        old = self._op_credit.get(op, 0.5)
        alpha = 0.08
        self._op_credit[op] = (1.0 - alpha) * old + alpha * reward

    def _replace_low_energy(self, shark_idx: int) -> None:
        if not self._unit_enabled("U05"):
            return

        if len(self.archive) >= 2:
            if self._unit_enabled("U07"):
                a = self._sample_archive().genotype
                b = self._sample_archive().genotype
                child = [a[j] if random.random() < 0.5 else b[j] for j in range(self.GENE_SIZE)]
                for j in range(self.GENE_SIZE):
                    if random.random() < 0.15:
                        child[j] = random.choice(self.GENE_VALUES)
                arch = child
            else:
                arch = self._scout()
        else:
            arch = self._scout()

        result = self._evaluate_arch(arch)
        self._archive_update(result)
        self.population[shark_idx] = result
        self._energy[shark_idx] = max(self.e0 * 0.5, self.e_min)

    def _accept(
        self,
        current: Individual,
        candidate: Individual,
        archive_improved: bool,
        progress: float,
    ) -> bool:
        if not self._unit_enabled("U04"):
            return True
        if dominates(candidate, current, self.evaluator.objective_directions):
            return True
        if archive_improved and self._unit_enabled("U07"):
            return True
        if not dominates(current, candidate, self.evaluator.objective_directions):
            accept_prob = 0.4 * (1.0 - 0.7 * progress)
            return random.random() < accept_prob
        return False

    def _active_base_probs(self) -> dict[str, float]:
        probs = dict(self.base_probs)
        if not self._unit_enabled("U02"):
            probs.pop("crossover", None)
        return self._normalize_probs(probs)

    def run(self) -> list[Individual]:
        self.population = []
        self.archive = []
        self._energy = []
        self._archive_sampler_dirty = True
        self._archive_sampler_weights = []
        self.evaluations = 0
        self.generations = 0

        for _ in range(self.pop_size):
            arch = self._random_arch()
            ind = self._evaluate_arch(arch)
            self.population.append(ind)
            self._energy.append(self.e0)
            self._archive_update(ind)

        if not self.population:
            return []

        self._truncate_archive()
        self.history.record(
            generation=self.generations,
            evaluations=self.evaluations,
            population=self.population,
            pareto_front=self.archive,
        )

        recent_improvements: list[int] = []

        total_iters = max(1, int(self.max_iterations))
        for k in range(1, total_iters + 1):

            archive_change_count = 0
            improvement_trials = 0
            progress = min(1.0, k / total_iters)

            if recent_improvements:
                imp_rate = sum(recent_improvements) / len(recent_improvements)
            else:
                imp_rate = 0.2
            probs = self._active_base_probs()
            if self._unit_enabled("U06"):
                probs = self._operator_probability_adaptation(probs, imp_rate, progress)
            else:
                probs = self._fixed_probability_schedule(progress)
            probs = self._credit_biased_probs(probs, progress)

            for i in range(len(self.population)):
                shark = self.population[i]
                op = self._choose_operator(probs)

                if op == "patrol":
                    candidate_arch = self._patrol(shark.genotype)
                elif op == "scent":
                    candidate_arch = self._scent(shark.genotype)
                elif op == "circle":
                    candidate_arch = self._circle(shark.genotype)
                elif op == "burst":
                    candidate_arch = self._burst(shark.genotype)
                elif op == "crossover":
                    candidate_arch = self._crossover(shark.genotype)
                else:
                    candidate_arch = self._scout()

                if candidate_arch == shark.genotype:
                    idx = random.randrange(self.GENE_SIZE)
                    candidate_arch = self._mutate_gene(candidate_arch, idx)

                candidate = self._evaluate_arch(candidate_arch)
                archive_improved = self._archive_update(candidate)
                if not self._unit_enabled("U03"):
                    self._truncate_archive()
                dominates_current = dominates(candidate, shark, self.evaluator.objective_directions)
                accepted = self._accept(shark, candidate, archive_improved, progress)

                if accepted:
                    energy_bonus = 0.2 if archive_improved else 0.05
                    self.population[i] = candidate
                    if self._unit_enabled("U05"):
                        self._energy[i] = min(self.e_max, self._energy[i] + energy_bonus)
                else:
                    if self._unit_enabled("U05"):
                        self._energy[i] = max(0.0, self._energy[i] - self.delta)

                if archive_improved:
                    archive_change_count += 1
                improvement_trials += 1
                self._update_operator_credit(op, accepted, archive_improved, dominates_current)

                if self._unit_enabled("U05") and self._energy[i] < self.e_min:
                    self._replace_low_energy(i)

            self._truncate_archive()

            recent_improvements.append(1 if (improvement_trials > 0 and archive_change_count > 0) else 0)
            if len(recent_improvements) > 20:
                recent_improvements.pop(0)

            self.generations = k
            self.history.record(
                generation=self.generations,
                evaluations=self.evaluations,
                population=self.population,
                pareto_front=self.archive,
            )

        return [self._clone_individual(ind) for ind in self.archive]


class MOSHOEnhancedSearch(MOSHOSearch):
    """MOSHO with clustered archive sampling and robustness safeguards."""
    SUPPORTED_UNITS = MOSHOSearch.SUPPORTED_UNITS | frozenset({"U08", "U10", "U11"})

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Evaluator,
        pop_size: int = 50,
        max_iterations: int = 300,
        archive_size: int | None = None,
        e0: float = 0.818,
        e_min: float = 0.508,
        e_max: float = 1.699,
        eta: float = 15.0,
        delta: float = 0.0968,
        history: History | None = None,
        cluster_threshold: int = 100000,
        kmeans_min_archive: int = 20,
        kmeans_k: int | None = None,
        kmeans_max_iter: int = 12,
        kmeans_refresh: int = 4,
        latency_stagnation_eps: float = 1e-3,
        latency_excellent_quantile: float = 0.1,
        mono_patience: int = 3,
        mono_refresh_interval: int = 12,
        disabled_units: set[str] | None = None,
    ):
        super().__init__(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=pop_size,
            max_iterations=max_iterations,
            archive_size=archive_size,
            e0=e0,
            e_min=e_min,
            e_max=e_max,
            eta=eta,
            delta=delta,
            disabled_units=disabled_units,
            history=history,
        )

        self.cluster_threshold = cluster_threshold
        self.kmeans_min_archive = kmeans_min_archive
        self.kmeans_k = kmeans_k
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_refresh = kmeans_refresh

        self.latency_stagnation_eps = latency_stagnation_eps
        self.latency_excellent_quantile = latency_excellent_quantile
        self.mono_patience = mono_patience
        self.mono_refresh_interval = mono_refresh_interval

        self._mono_mode: int | None = None
        self._mono_active = False
        self._mono_counter = 0
        self._mono_refresh_counter = 0

        self._cluster_dirty = True
        self._cluster_refresh_counter = 0
        self._cluster_members: list[list[int]] | None = None

        self._sampler_refresh_generation = -1

        self._use_cluster_sampling = (
            self._estimate_search_space_size() >= self.cluster_threshold
        )

    def _estimate_search_space_size(self) -> int:
        try:
            return int(self.search_space.num_ops) ** int(self.search_space.num_edges)
        except Exception:
            return 0

    def _archive_update(self, candidate: Individual) -> bool:
        changed = super()._archive_update(candidate)
        if changed:
            self._cluster_dirty = True
        return changed

    def _truncate_archive(self) -> None:
        if len(self.archive) <= self.archive_size:
            return

        if not self._unit_enabled("U10"):
            super()._truncate_archive()
            self._cluster_dirty = True
            return

        n_obj = len(self.evaluator.objective_directions)
        scores = [0.0 for _ in self.archive]
        extreme_idx: set[int] = set()

        for obj in range(n_obj):
            direction = self.evaluator.objective_directions[obj]
            order = sorted(
                range(len(self.archive)),
                key=lambda i: self.archive[i].fitness[obj] * direction,
            )
            if not order:
                continue

            low_idx = order[0]
            high_idx = order[-1]
            extreme_idx.add(low_idx)
            extreme_idx.add(high_idx)

            scores[low_idx] = float("inf")
            scores[high_idx] = float("inf")

            min_v = self.archive[low_idx].fitness[obj] * direction
            max_v = self.archive[high_idx].fitness[obj] * direction
            if max_v == min_v:
                continue

            for k in range(1, len(order) - 1):
                idx = order[k]
                if scores[idx] == float("inf"):
                    continue
                prev_v = self.archive[order[k - 1]].fitness[obj] * direction
                next_v = self.archive[order[k + 1]].fitness[obj] * direction
                scores[idx] += (next_v - prev_v) / (max_v - min_v)

        order = sorted(range(len(self.archive)), key=lambda i: scores[i], reverse=True)
        if len(extreme_idx) >= self.archive_size:
            keep_idx = set(sorted(extreme_idx, key=lambda i: scores[i], reverse=True)[: self.archive_size])
        else:
            keep_idx = set(extreme_idx)
            for idx in order:
                if idx in keep_idx:
                    continue
                keep_idx.add(idx)
                if len(keep_idx) >= self.archive_size:
                    break

        self.archive = [self.archive[i] for i in range(len(self.archive)) if i in keep_idx]
        self._archive_sampler_dirty = True
        self._cluster_dirty = True

    def _refresh_cluster_cache(self) -> None:
        if not self._use_cluster_sampling:
            self._cluster_members = None
            return
        if len(self.archive) < self.kmeans_min_archive:
            self._cluster_members = None
            return

        if not self._cluster_dirty and self._cluster_refresh_counter > 0:
            self._cluster_refresh_counter -= 1
            return

        fitness = np.array([ind.fitness for ind in self.archive], dtype=float)
        if fitness.size == 0:
            self._cluster_members = None
            return

        directions = np.array(self.evaluator.objective_directions, dtype=float)
        fitness = fitness * directions

        mins = fitness.min(axis=0)
        maxs = fitness.max(axis=0)
        denom = np.where(maxs - mins > 0, maxs - mins, 1.0)
        norm = (fitness - mins) / denom

        n = norm.shape[0]
        k = self.kmeans_k or max(2, int(round(math.sqrt(n))))
        k = min(k, n)
        if k < 2:
            self._cluster_members = None
            return

        labels = self._kmeans(norm, k)
        clusters = [[] for _ in range(k)]
        for idx, label in enumerate(labels):
            clusters[int(label)].append(idx)

        self._cluster_members = [c for c in clusters if c]
        self._cluster_dirty = False
        self._cluster_refresh_counter = self.kmeans_refresh

    def _kmeans(self, data: np.ndarray, k: int) -> np.ndarray:
        n = data.shape[0]
        init_idx = random.sample(range(n), k)
        centroids = data[init_idx].copy()
        labels = np.zeros(n, dtype=int)

        for _ in range(self.kmeans_max_iter):
            dists = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                members = data[labels == c]
                if len(members) == 0:
                    centroids[c] = data[random.randrange(n)]
                else:
                    centroids[c] = members.mean(axis=0)
        return labels

    def _sample_archive(self) -> Individual:
        if not self.archive:
            raise ValueError("Cannot sample from an empty archive.")

        if not self._unit_enabled("U08"):
            return random.choice(self.archive)

        needs_refresh = (
            self._archive_sampler_dirty
            or len(self._archive_sampler_weights) != len(self.archive)
        )
        if needs_refresh and self.generations != self._sampler_refresh_generation:
            self._refresh_archive_sampler()
            self._sampler_refresh_generation = self.generations

        weights_ok = len(self._archive_sampler_weights) == len(self.archive)

        self._refresh_cluster_cache()
        if not self._cluster_members:
            if weights_ok:
                return random.choices(self.archive, weights=self._archive_sampler_weights, k=1)[0]
            return random.choice(self.archive)

        cluster = random.choice(self._cluster_members)
        if weights_ok:
            weights = [self._archive_sampler_weights[i] for i in cluster]
            idx = random.choices(cluster, weights=weights, k=1)[0]
        else:
            idx = random.choice(cluster)
        return self.archive[idx]

    def _update_mono_objective_mode(self) -> None:
        if len(self.archive) < 3:
            self._mono_mode = None
            self._mono_active = False
            self._mono_counter = 0
            return

        dirs = self.evaluator.objective_directions
        if len(dirs) < 2:
            self._mono_mode = None
            self._mono_active = False
            self._mono_counter = 0
            return

        lat_idx = 1
        lats = [ind.fitness[lat_idx] for ind in self.archive if ind.fitness is not None]
        if not lats:
            self._mono_mode = None
            self._mono_active = False
            self._mono_counter = 0
            return

        lat_range = max(lats) - min(lats)
        if lat_range <= self.latency_stagnation_eps:
            q = float(np.quantile(lats, self.latency_excellent_quantile))
            if min(lats) <= q:
                acc_idx = next((i for i, d in enumerate(dirs) if d > 0), 0)
                if self._mono_mode == acc_idx:
                    self._mono_counter += 1
                else:
                    self._mono_mode = acc_idx
                    self._mono_counter = 1
                self._mono_active = self._mono_counter >= self.mono_patience
                return

        self._mono_mode = None
        self._mono_active = False
        self._mono_counter = 0

    def _accept(
        self,
        current: Individual,
        candidate: Individual,
        archive_improved: bool,
        progress: float,
    ) -> bool:
        if not self._unit_enabled("U11"):
            return super()._accept(current, candidate, archive_improved, progress)

        if self._mono_refresh_counter <= 0:
            self._update_mono_objective_mode()
            self._mono_refresh_counter = self.mono_refresh_interval
        else:
            self._mono_refresh_counter -= 1

        if self._mono_active and self._mono_mode is not None:
            obj = self._mono_mode
            direction = self.evaluator.objective_directions[obj]
            if candidate.fitness[obj] * direction > current.fitness[obj] * direction:
                return True
            if archive_improved:
                return True
            accept_prob = 0.2 * (1.0 - 0.7 * progress)
            return random.random() < accept_prob

        return super()._accept(current, candidate, archive_improved, progress)


class GeneticAlgorithmNG(SearchStrategy):
    """NSGA-III/NG style: hybrid NG search strategy (Liu Yuejun, Heliyon 2024).

    At each generation, parent selection is decided probabilistically:
      - With prob p_neigh  → NeighborSelection (Eq. 3)
      - With prob p_guide  → GuidanceSelection (Eq. 4)
      - Otherwise          → fallback TournamentSelection

    Paper defaults: p_neigh=0.8, p_guide=0.2, k=3, P=13 (for NSGA-III variant).
    """

    def __init__(self, population: Population, selection: Selection,
                 crossover: Crossover, mutation: Mutation,
                 replacement: Replacement, evaluator: Evaluator,
                 budget: int = 500,
                 p_neigh: float = 0.8,
                 p_guide: float = 0.2,
                 k: int = 3,
                 P: int = 13,
                 termination: Termination | None = None,
                 history: History | None = None):
        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(population, selection, variation, replacement,
                         evaluator, termination=termination,
                         history=history, budget=budget)
        self.p_neigh = p_neigh
        self.p_guide = p_guide
        self._neighbor_sel = NeighborSelection(k=k, P=P)
        self._guidance_sel = GuidanceSelection()
        self._fallback_sel = selection  # tournament

    def _select_parents(self) -> list[Individual]:
        r = random.random()
        inds = self.population.individuals
        dirs = self.evaluator.objective_directions
        n = self.population.size
        if r < self.p_neigh:
            return self._neighbor_sel.select(inds, n, dirs)
        elif r < self.p_neigh + self.p_guide:
            return self._guidance_sel.select(inds, n, dirs)
        else:
            return self._fallback_sel.select(inds, n, dirs)

    def run(self) -> Population:
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            parents = self._select_parents()
            offspring = self.variation.generate(parents, self.population.size)
            self._evaluate_offspring(offspring)
            self.population.individuals = self.replacement.replace(
                self.population.individuals,
                offspring,
                self.population.size,
                self.evaluator.objective_directions,
            )
            self.generations += 1
            self._record_history()

        return self.population


class ABCSearchStrategy(SearchStrategy):
    """Artificial Bee Colony NAS search strategy.

    Implements the HiveNAS framework (Shahawy & Benkhelifa, arXiv:2211.10250v2).
    The algorithm follows Algorithm 1 of the paper:

        Initialisation (Scout phase)
        FOR t = 1, 2, ..., T:
            Employee Bees  – evaluate current source; sample 1-op neighbor;
                             greedy selection (keep best).
            Onlooker Bees  – self.selection (RouletteWheelSelection) assigns
                             onlookers to food sources; sample neighbor;
                             greedy selection.
            Scout Bees     – reset any source whose trial_count >= limit.
        Output best candidate.

    Parameters
    ----------
    population : ABCPopulation
        Carries the FoodSource list and the abandonment_limit.
    neighbor_sampler : ABCNeighborSampler
        Generates 1-operation neighbors in the discrete NAS space.
    selection : RouletteWheelSelection
        Fitness-proportionate selector used in the onlooker phase.
        Passed straight to the base class as self.selection.
    evaluator : Evaluator
        Queries accuracy and latency from the benchmark.
    budget : int
        Total evaluation budget (stopping criterion).
    termination, history :
        Optional overrides for the base class.
    """

    def __init__(self,
                 population,
                 neighbor_sampler,
                 selection: Selection,
                 evaluator: Evaluator,
                 budget: int = 500,
                 termination: Termination | None = None,
                 history: History | None = None,
                 dmt_fraction: float = 0.67):

        from nas_framework.variation import MutationOnlyVariation
        from nas_framework.mutation import SinglePointMutation
        from nas_framework.replacement import ElitistReplacement

        variation   = MutationOnlyVariation(SinglePointMutation(population.search_space))
        replacement = ElitistReplacement()

        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            termination=termination,
            history=history,
            budget=budget,
        )
        self.neighbor_sampler = neighbor_sampler
        # dmt: scout resets disabled after this many evaluations (MBO-ABCFE §3.2).
        # Prevents random restarts from destroying good late-search solutions.
        self._dmt = int(dmt_fraction * budget)
        # Visited cache: genotype tuple -> fitness tuple.
        # Prevents re-spending evaluations on already-seen architectures.
        self._visited: dict[tuple, tuple] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_individual(self, individual: Individual) -> None:
        """Evaluate an individual in-place and increment the counter.

        Skips re-evaluation if the genotype was already seen (visited cache),
        recovering wasted budget from duplicate neighbor samples.
        At large budgets (500+), 20-39% of neighbor samples are duplicates
        without this guard.
        
        IMPORTANT: Evaluations counter is incremented on EVERY call (including 
        cache hits), not just new architectures. This ensures the termination 
        condition can be met even when the search saturates the neighborhood.
        """
        key = tuple(individual.genotype)
        if key in self._visited:
            # Retrieve cached fitness without spending a new evaluation call.
            individual.fitness = self._visited[key]
            if hasattr(self.population.search_space, "metadata_from_genotype"):
                individual.metadata = (
                    self.population.search_space.metadata_from_genotype(individual.genotype)
                )
        else:
            # New architecture - evaluate and cache it
            individual.fitness = self.evaluator.evaluate(individual.genotype)
            if hasattr(self.population.search_space, "metadata_from_genotype"):
                individual.metadata = (
                    self.population.search_space.metadata_from_genotype(individual.genotype)
                )
            self._visited[key] = individual.fitness
        
        # CRITICAL FIX: Increment on EVERY _eval_individual attempt.
        # This ensures the evaluation budget and termination condition work correctly
        # even when the search saturates and samples mostly duplicate neighbors.
        self.evaluations += 1

    def _employee_phase(self) -> None:
        """Each employee evaluates a 1-op neighbor; greedy keep-best."""
        dirs = self.evaluator.objective_directions
        for fs in self.population.food_sources:
            if self.termination.should_stop(self.evaluations, self.generations):
                return
            neighbor = self.neighbor_sampler.sample_neighbor(fs.individual)
            self._eval_individual(neighbor)
            fs.update(neighbor, dirs)

    def _onlooker_phase(self) -> None:
        """Roulette-wheel assign onlookers to food sources; greedy keep-best."""
        dirs = self.evaluator.objective_directions
        n_onlookers = len(self.population.food_sources)

        self.population.sync_individuals()
        selected_inds = self.selection.select(
            self.population.individuals,
            n_onlookers,
            dirs,
        )

        # Build a genotype→FoodSource map for O(1) lookup.
        geno_to_fs: dict[tuple, object] = {
            tuple(fs.individual.genotype): fs
            for fs in self.population.food_sources
        }

        for ind in selected_inds:
            if self.termination.should_stop(self.evaluations, self.generations):
                return
            fs = geno_to_fs.get(tuple(ind.genotype))
            if fs is None:
                continue
            neighbor = self.neighbor_sampler.sample_neighbor(fs.individual)
            self._eval_individual(neighbor)
            fs.update(neighbor, dirs)

    def _scout_phase(self) -> None:
        """Reset exhausted sources — suppressed after dmt evaluations.

        Past the dmt threshold (default 67% of budget) the population has
        typically converged to a good region; random restarts would waste
        budget.  This mirrors the discard-mechanism-trigger from MBO-ABCFE.
        """
        if self.evaluations >= self._dmt:
            return  # scout suppressed in late search
        resets = self.population.scout_reset(self.evaluator.objective_directions)
        self.evaluations += resets

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Population:
        """Execute the full ABC search loop (Algorithm 1 of HiveNAS)."""
        self.population.initialize()
        self.evaluations = len(self.population.food_sources)
        self.generations = 0
        self.population.sync_individuals()
        # Seed visited cache with initial food sources.
        for fs in self.population.food_sources:
            if fs.individual.fitness is not None:
                self._visited[tuple(fs.individual.genotype)] = fs.individual.fitness
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            self._employee_phase()
            if self.termination.should_stop(self.evaluations, self.generations):
                break
            self._onlooker_phase()
            if self.termination.should_stop(self.evaluations, self.generations):
                break
            self._scout_phase()
            self.population.sync_individuals()
            self.generations += 1
            self._record_history()

        self.population.sync_individuals()
        return self.population


class FireflySearchStrategy(SearchStrategy):
    """Improved Firefly Algorithm for NAS (RB-IFA, Nguyen et al., ICAART 2025).

    Each individual (firefly) is attracted to brighter (better rank-scored)
    neighbours and moves toward them.  Brightness is determined by the
    rank-based scalar score from mo_utils.rank_based_score, which collapses
    multiple objectives into one weighted value — no Pareto front needed.

    Movement in the discrete NAS space is modelled as:
      - With probability proportional to attractiveness β(r): adopt one gene
        from the brighter firefly (single-gene crossover toward the target).
      - Otherwise: apply a random mutation (exploration / light absorption).

    When the population stagnates for *max_chances* generations, a genetic
    iteration (tournament selection + crossover + mutation on the full
    population) is triggered to escape local optima, mirroring the IFA
    mechanism of Mokhtari et al. (2022) used in the paper.

    Parameters
    ----------
    population : Population
    selection  : Selection   — used for the genetic fallback iteration.
    crossover  : Crossover   — used for the genetic fallback iteration.
    mutation   : Mutation    — used for both movement exploration and fallback.
    replacement: Replacement — RankBasedReplacement recommended.
    evaluator  : Evaluator
    budget     : int         — total evaluation budget.
    w_perf     : float       — performance weight for rank scoring (0–1).
    gamma      : float       — light absorption coefficient.
                              0 → every firefly sees global best (PSO-like).
                              Large → fireflies ignore distant ones (local search).
                              Default 1.0 (moderate, balanced).
    beta0      : float       — base attractiveness at distance 0. Default 1.0.
    max_chances: int         — stagnation patience before genetic fallback.
    """

    def __init__(self,
                 population: Population,
                 selection: Selection,
                 crossover: Crossover,
                 mutation: Mutation,
                 replacement: Replacement,
                 evaluator: Evaluator,
                 budget: int = 500,
                 w_perf: float = 0.6,
                 gamma: float = 1.0,
                 beta0: float = 1.0,
                 max_chances: int = 5,
                 use_fap: bool = True,
                 fa_prob: float = 0.5,
                 termination: Termination | None = None,
                 history: History | None = None):

        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            termination=termination,
            history=history,
            budget=budget,
        )
        self.w_perf      = w_perf
        self.gamma       = gamma
        self.beta0       = beta0
        self.max_chances = max_chances
        self.use_fap     = use_fap    # True → flat FAP gate; False → β(r) formula
        self.fa_prob     = fa_prob    # per-gene fire probability when use_fap=True
        self._mutation   = mutation
        self._crossover  = crossover

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hamming(self, a: Individual, b: Individual) -> int:
        """Hamming distance between two genotypes."""
        return sum(g1 != g2 for g1, g2 in zip(a.genotype, b.genotype))

    def _attractiveness(self, r: int) -> float:
        """β(r) = β0 * exp(-γ * r²)  — decreases with distance."""
        import math
        return self.beta0 * math.exp(-self.gamma * r * r)

    def _move_toward(self, firefly: Individual,
                     target: Individual) -> Individual:
        """Move firefly one step toward target.

        If use_fap=True: with probability fa_prob per gene, adopt target's gene.
        If use_fap=False: with probability β(r) = β0 * exp(-γ * r²), adopt target's gene.
        Remaining genes stay or receive a random mutation with prob α=0.2.
        """
        import random
        r   = self._hamming(firefly, target)
        
        if self.use_fap:
            beta = self.fa_prob
        else:
            beta = self._attractiveness(r)
            
        alpha = 0.2   # random walk component

        new_geno = list(firefly.genotype)
        for i, (g_self, g_target) in enumerate(
                zip(firefly.genotype, target.genotype)):
            if g_self != g_target and random.random() < beta:
                new_geno[i] = g_target          # attracted move
            elif random.random() < alpha:
                # random walk: re-sample gene
                n_ops = self.population.search_space.num_ops
                new_geno[i] = random.randint(0, n_ops - 1)
        return Individual(new_geno)

    def _scores(self) -> list[float]:
        from nas_framework.mo_utils import rank_based_score
        return rank_based_score(
            self.population.individuals,
            self.evaluator.objective_directions,
            self.w_perf,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Population:
        """Execute the RB-IFA search loop."""
        import random

        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations  = 0
        self._record_history()

        best_score    = min(self._scores())
        chances       = self.max_chances

        while not self.termination.should_stop(self.evaluations, self.generations):
            scores = self._scores()
            inds   = self.population.individuals

            # ── Firefly movement phase ────────────────────────────────
            new_inds: list[Individual] = []
            for i, fi in enumerate(inds):
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                # Find a brighter firefly (lower score = brighter).
                brighter = [inds[j] for j, s in enumerate(scores)
                            if s < scores[i]]
                if brighter:
                    target  = random.choice(brighter)
                    moved   = self._move_toward(fi, target)
                else:
                    # Already brightest — random walk only.
                    moved = self._mutation.mutate(fi)

                moved.fitness = self.evaluator.evaluate(moved.genotype)
                if hasattr(self.population.search_space, "metadata_from_genotype"):
                    moved.metadata = (
                        self.population.search_space
                        .metadata_from_genotype(moved.genotype)
                    )
                self.evaluations += 1
                new_inds.append(moved)

            # ── Rank-based survivor selection ─────────────────────────
            self.population.individuals = self.replacement.replace(
                self.population.individuals,
                new_inds,
                self.population.size,
                self.evaluator.objective_directions,
            )

            # ── Stagnation check ──────────────────────────────────────
            current_best = min(self._scores())
            if current_best < best_score:
                best_score = current_best
                chances    = self.max_chances
            else:
                chances -= 1

            # ── Genetic fallback iteration ────────────────────────────
            if chances == 0:
                parents  = self.selection.select(
                    self.population.individuals,
                    self.population.size,
                    self.evaluator.objective_directions,
                )
                offspring = self.variation.generate(parents, self.population.size)
                self._evaluate_offspring(offspring)
                self.population.individuals = self.replacement.replace(
                    self.population.individuals,
                    offspring,
                    self.population.size,
                    self.evaluator.objective_directions,
                )
                # Reset rank scores after genetic refresh.
                best_score = min(self._scores())
                chances    = self.max_chances

            self.generations += 1
            self._record_history()

        return self.population

class HybridMBOStrategy(SearchStrategy):
    """Rank-stratified MBO hybrid: FA movement on top half, GA on bottom half.

    Each generation the population is ranked by rank_based_score().
    - SP1 (top rank_fraction, default 50%): each individual moves toward a
      random SP1 neighbour using the flat FAP gate (adopt gene if FAP < fa_prob).
    - SP2 (bottom half): standard tournament selection + crossover + mutation
      (same as GeneticAlgorithm).
    Both halves are merged and the best `pop_size` survivors are kept via
    RankBasedReplacement.

    This directly implements the SP1/SP2 idea from the MBO resume (§6.3):
    FA as default exploitation on good individuals, GA as diversification
    on weaker ones — no stagnation counter needed.

    Parameters
    ----------
    population   : Population
    selection    : Selection      (TournamentSelection for GA phase)
    crossover    : Crossover
    mutation     : Mutation       (SinglePointMutation)
    replacement  : Replacement    (RankBasedReplacement recommended)
    evaluator    : Evaluator
    budget       : int
    w_perf       : float          rank weight for accuracy (default 0.6)
    rank_fraction: float          fraction of pop in SP1 (default 0.5)
    fa_prob      : float          per-gene FAP gate probability (default 0.5)
    """

    def __init__(self, population, selection, crossover, mutation,
                 replacement, evaluator, budget: int = 500,
                 termination=None, history=None,
                 w_perf: float = 0.6,
                 rank_fraction: float = 0.5,
                 fa_prob: float = 0.5):
        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            budget=budget,
            termination=termination,
            history=history,
        )
        self.w_perf       = w_perf
        self.rank_fraction = rank_fraction
        self.fa_prob      = fa_prob
        self._mutation    = mutation
        self._crossover   = crossover

    def _rank_scores(self):
        from nas_framework.mo_utils import rank_based_score
        return rank_based_score(
            self.population.individuals,
            self.evaluator.objective_directions,
            w_perf=self.w_perf,
        )

    def _fap_move(self, ind: Individual, sp1: list) -> Individual:
        """Move ind toward a random SP1 member using flat FAP gate."""
        target   = random.choice(sp1)
        new_geno = list(ind.genotype)
        for i, (g_self, g_tgt) in enumerate(zip(ind.genotype, target.genotype)):
            if g_self != g_tgt and random.random() < self.fa_prob:
                new_geno[i] = g_tgt
        return Individual(new_geno)

    def run(self) -> Population:
        self.population.initialize()
        self._evaluate_offspring(self.population.individuals)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            scores = self._rank_scores()
            order  = sorted(range(len(scores)), key=lambda i: scores[i])
            n_sp1  = max(1, int(self.rank_fraction * self.population.size))
            sp1_idx = order[:n_sp1]
            sp2_idx = order[n_sp1:]
            inds    = self.population.individuals
            sp1     = [inds[i] for i in sp1_idx]

            offspring: list[Individual] = []

            # ── SP1: FA movement ──────────────────────────────────────
            for idx in sp1_idx:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                moved = self._fap_move(inds[idx], sp1)
                offspring.append(moved)

            # ── SP2: GA (crossover + mutation) ────────────────────────
            if sp2_idx and not self.termination.should_stop(
                    self.evaluations, self.generations):
                sp2_inds = [inds[i] for i in sp2_idx]
                ga_offspring = self.variation.generate(sp2_inds, len(sp2_idx))
                offspring.extend(ga_offspring)

            self._evaluate_offspring(offspring)

            # ── Rank-based survivor selection ─────────────────────────
            self.population.individuals = self.replacement.replace(
                self.population.individuals,
                offspring,
                self.population.size,
                self.evaluator.objective_directions,
            )

            self.generations += 1
            self._record_history()

        return self.population


class PSOSearchStrategy(SearchStrategy):
    """MOIPSO: Multi-Objective PSO with trigonometric acceleration and
    adaptive Gaussian mutation 

    Algorithm per iteration
    -----------------------
    1. Draw dynamic acceleration factors:
           c1 = 2.05 * |cos(2*pi*rand)|
           c2 = 2.05 * |sin(2*pi*rand)|
    2. Update each particle's velocity and position (discrete PSO, Eq. 5-6).
    3. Evaluate new positions.
    4. Apply GaussianMutation (sigma = 0.1*(1 - t/T)) to a random half of
       the population; evaluate mutated particles.
    5. CrowdingReplacement: keep best *pop_size* by Pareto rank + crowding.
    6. Update pbests and gbest.

    Parameters
    ----------
    population : PSOPopulation
    selection  : Selection      (kept for signature compatibility; not used
                                 in main PSO loop — gbest drives movement)
    crossover  : Crossover      (same — not used in main loop)
    mutation   : GaussianMutation
    replacement: CrowdingReplacement  (or any Replacement)
    evaluator  : Evaluator
    budget     : int            (total number of architecture evaluations)
    w          : float          (inertia weight; default 0.4)
    """

    def __init__(self, population, selection, crossover, mutation,
                 replacement, evaluator, budget: int = 500,
                 termination=None, history=None, w: float = 0.4):
        # Build a dummy variation so SearchStrategy.__init__ is satisfied
        from nas_framework.variation import CrossoverMutationVariation
        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            budget=budget,
            termination=termination,
            history=history,
        )
        self.w = w
        self._mutation = mutation     # GaussianMutation (direct access)
        self.budget = budget

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trig_coefficients(self):
        """Return dynamic c1, c2 (Eq. 8 from MOIPSO paper)."""
        import math
        r = random.random()
        c1 = 2.05 * abs(math.cos(2 * math.pi * r))
        c2 = 2.05 * abs(math.sin(2 * math.pi * r))
        return c1, c2

    def _evaluate_particle(self, ind):
        """Evaluate a single Individual, update evaluations counter."""
        ind.fitness = self.evaluator.evaluate(ind.genotype)
        if hasattr(self.population.search_space, "metadata_from_genotype"):
            ind.metadata = self.population.search_space.metadata_from_genotype(
                ind.genotype)
        self.evaluations += 1

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Execute the MOIPSO search loop and return the final Population."""
        # Wire GaussianMutation's progress function to this strategy's state
        if hasattr(self._mutation, '_get_progress'):
            self._mutation._get_progress = lambda: self.evaluations / max(1, self.budget)

        # Also set the PSOPopulation's inertia weight from strategy param
        self.population.w = self.w

        # Step 1 — Initialise (random positions, zero velocities, evaluate)
        self.population.initialize()
        self.evaluations = len(self.population.particles)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            # Step 2 — Dynamic acceleration factors
            c1, c2 = self._trig_coefficients()

            # Step 3 — Velocity + position update (discrete PSO)
            self.population.update_velocities_and_positions(c1, c2)

            # Step 4 — Evaluate new positions
            for p in self.population.particles:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                self._evaluate_particle(p.individual)

            # Step 5 — Adaptive Gaussian mutation on random half of particles
            n_mutate = max(1, len(self.population.particles) // 2)
            targets = random.sample(self.population.particles, n_mutate)
            for p in targets:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                mutated = self._mutation.mutate(p.individual)
                self._evaluate_particle(mutated)
                # Greedy replacement: keep mutated if better
                from nas_framework.population import _weighted_score
                dirs = self.evaluator.objective_directions
                if (mutated.fitness is not None and
                    (p.individual.fitness is None or
                     _weighted_score(mutated.fitness, dirs) >
                     _weighted_score(p.individual.fitness, dirs))):
                    p.individual = mutated

            # Step 6 — CrowdingReplacement: merge old + new, keep best N
            self.population.sync_individuals()
            # Filter out any individuals whose fitness was not set (e.g. if
            # budget was exhausted mid-loop before evaluation completed).
            alive = [ind for ind in self.population.individuals
                     if ind.fitness is not None]
            if not alive:
                alive = self.population.individuals  # fallback: keep all
            self.population.individuals = self.replacement.replace(
                alive,
                [],   # offspring already merged into particles above
                self.population.size,
                self.evaluator.objective_directions,
            )

            # Step 7 — Update pbests and gbest
            self.population.update_pbests()

            self.generations += 1
            self._record_history()

        return self.population

class MBOSearchStrategy(SearchStrategy):
    """Monarch Butterfly Optimization hybridized with ABC exploration and
    Firefly exploitation (MBO-ABCFE), adapted for multi-objective NAS.

    Population is split each generation into SP1 (top rank_fraction) and
    SP2 (remainder) by rank_based_score().

    SP1 update (for each individual):
        With prob MR:
            FAP ~ Uniform(0,1) per gene
            if FAP < fa_prob  → FA attraction toward a random SP1 neighbor
            else              → MBO migration (copy gene from SP1 or SP2)
        After update: if old individual was better, increment trial counter.

    SP2 update (for each individual):
        For each gene:
            if rand < p  → copy gene from current best (butterfly adjust)
            else         → copy gene from random SP2 member
            if rand > BAR → apply single-point mutation (Lévy flight proxy)
        After update: if old individual was better, increment trial counter.

    Scout phase (ABC abandonment):
        If evaluations < dmt_fraction * budget:
            Replace individuals with trial >= exh by fresh random solutions.

    Parameters
    ----------
    population      : Population (standard Population, not ABCPopulation)
    selection       : Selection  (unused in main loop; kept for API compat)
    crossover       : Crossover  (unused in main loop; kept for API compat)
    mutation        : Mutation   (SinglePointMutation — Lévy proxy)
    replacement     : Replacement (RankBasedReplacement recommended)
    evaluator       : Evaluator
    budget          : int
    w_perf          : float  rank weight for accuracy objective (default 0.6)
    rank_fraction   : float  fraction of pop assigned to SP1 (default 0.42)
    p               : float  migration ratio — gene drawn from SP1 if rand<p
    BAR             : float  butterfly adjusting rate (Lévy gate, default 5/12)
    MR              : float  modification rate — FA/migration gate (default 0.8)
    fa_prob         : float  per-gene FA fire probability (FAP gate, default 0.5)
    exh_fraction    : float  trial limit = exh_fraction * (budget / pop_size)
    dmt_fraction    : float  scout suppressed after this fraction of budget
    """

    def __init__(self, population, selection, crossover, mutation,
                 replacement, evaluator, budget: int = 500,
                 termination=None, history=None,
                 w_perf: float = 0.8,
                 rank_fraction: float = 0.42,
                 p: float = 5/12,
                 BAR: float = 0.9,
                 MR: float = 0.8,
                 fa_prob: float = 0.5,
                 exh_fraction: float = 4.0,
                 dmt_fraction: float = 0.67):
        from nas_framework.variation import CrossoverMutationVariation
        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            budget=budget,
            termination=termination,
            history=history,
        )
        self.w_perf        = w_perf
        self.rank_fraction = rank_fraction
        self.p             = p
        self.BAR           = BAR
        self.MR            = MR
        self.fa_prob       = fa_prob
        self._mutation     = mutation
        self.budget        = budget

        # Compute exh and dmt analogously to paper formulae
        pop_size           = population.size
        # exh = round(maxIter / Np * exh_fraction); maxIter ≈ budget/pop_size
        max_iter_approx    = max(1, budget // pop_size)
        # exh_fraction used to scale abandonment limit (limit = approx_iters / fraction)
        # We want individuals to be abandoned if they don't improve for ~1/4 of the run.
        self.exh = max(3, round(max_iter_approx / exh_fraction))
        self.dmt = round(dmt_fraction * budget)   # in evaluations, not iters

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rank_scores(self):
        from nas_framework.mo_utils import rank_based_score
        return rank_based_score(
            self.population.individuals,
            self.evaluator.objective_directions,
            w_perf=self.w_perf,
        )

    def _eval(self, ind):
        ind.fitness = self.evaluator.evaluate(ind.genotype)
        if hasattr(self.population.search_space, "metadata_from_genotype"):
            ind.metadata = self.population.search_space.metadata_from_genotype(
                ind.genotype)
        self.evaluations += 1

    def _is_better(self, a, b):
        """Return True if individual a is better than b (lower rank = better)."""
        from nas_framework.population import _weighted_score
        dirs = self.evaluator.objective_directions
        if a.fitness is None: return False
        if b.fitness is None: return True
        
        # Apply w_perf locally to ensure greedy selection respects accuracy focus
        # This avoiding touching global _weighted_score as requested.
        w_acc = self.w_perf
        w_lat = 1.0 - w_acc
        
        score_a = (a.fitness[0] * dirs[0] * w_acc) + (a.fitness[1] * dirs[1] * w_lat)
        score_b = (b.fitness[0] * dirs[0] * w_acc) + (b.fitness[1] * dirs[1] * w_lat)
        return score_a > score_b

    def _fa_move(self, ind, sp1):
        """FA attraction: move ind toward a random brighter SP1 member gene-by-gene."""
        from copy import deepcopy
        if not sp1:
            return self._mutation.mutate(ind)
        target = random.choice(sp1)
        geno   = deepcopy(ind.genotype)
        for j in range(len(geno)):
            if random.random() < self.fa_prob:
                geno[j] = target.genotype[j]
        return Individual(geno)

    def _mbo_migrate(self, ind, sp1, sp2):
        """MBO migration: copy each gene from SP1 or SP2 based on ratio p."""
        from copy import deepcopy
        geno = deepcopy(ind.genotype)
        sp1_list = list(sp1)
        sp2_list = list(sp2)
        for j in range(len(geno)):
            r = random.random() * 1.2   # peri = 1.2 as in paper
            if r <= self.p and sp1_list:
                donor = random.choice(sp1_list)
            elif sp2_list:
                donor = random.choice(sp2_list)
            else:
                continue
            geno[j] = donor.genotype[j]
        return Individual(geno)

    def _butterfly_adjust(self, ind, best, sp2):
        """SP2 butterfly adjusting: copy best or random SP2 + optional Lévy."""
        from copy import deepcopy
        geno     = deepcopy(ind.genotype)
        sp2_list = list(sp2)
        n_genes  = len(geno)
        for j in range(n_genes):
            r = random.random() * 1.2
            if r <= self.p and best is not None:
                geno[j] = best.genotype[j]
            elif sp2_list:
                donor  = random.choice(sp2_list)
                geno[j] = donor.genotype[j]
            # Lévy flight proxy: BAR gate — if BAR is small, mutation is rare.
            # We now apply this only with probability 1/n_genes to avoid random walk.
            if random.random() > self.BAR:
                choices = [op for op in
                           range(self.population.search_space.num_ops)
                           if op != geno[j]]
                if choices:
                    geno[j] = random.choice(choices)
        return Individual(geno)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        # Initialise
        self.population.initialize()
        trials = [0] * self.population.size
        for ind in self.population.individuals:
            self._eval(ind)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            # ── Split by rank ──────────────────────────────────────────
            scores   = self._rank_scores()
            order    = sorted(range(len(scores)), key=lambda i: scores[i])
            n_sp1    = max(1, int(self.rank_fraction * self.population.size))
            sp1_idx  = set(order[:n_sp1])
            sp2_idx  = set(order[n_sp1:])
            inds     = self.population.individuals
            sp1      = [inds[i] for i in sp1_idx]
            sp2      = [inds[i] for i in sp2_idx]
            best     = inds[order[0]] if order else None

            new_inds = list(inds)   # mutable copy

            # ── SP1 update ─────────────────────────────────────────────
            for idx in sp1_idx:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                old = inds[idx]
                theta = random.random()
                if theta <= self.MR:
                    candidate = self._fa_move(old, sp1)
                else:
                    candidate = self._mbo_migrate(old, sp1, sp2)
                self._eval(candidate)
                if self._is_better(candidate, old):
                    new_inds[idx] = candidate
                    trials[idx]   = 0
                else:
                    trials[idx]  += 1

            # ── SP2 update ─────────────────────────────────────────────
            for idx in sp2_idx:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                old       = inds[idx]
                candidate = self._butterfly_adjust(old, best, sp2)
                self._eval(candidate)
                if self._is_better(candidate, old):
                    new_inds[idx] = candidate
                    trials[idx]   = 0
                else:
                    trials[idx]  += 1

            # ── Scout phase (ABC abandonment — early search only) ──────
            if self.evaluations < self.dmt:
                for idx in range(self.population.size):
                    if trials[idx] >= self.exh:
                        if self.termination.should_stop(
                                self.evaluations, self.generations):
                            break
                        fresh = Individual(
                            self.population.search_space.random_individual())
                        self._eval(fresh)
                        new_inds[idx] = fresh
                        trials[idx]   = 0

            # ── Update population ──────────────────────────────────────
            self.population.individuals = new_inds

            self.generations += 1
            self._record_history()

        return self.population

class APSOESearch:
    # =============================================================================
# APSO-E: Archive-Guided PSO with Elite Perturbation
# =============================================================================
# Bottlenecks diagnosed in ABC / Firefly / PSO / HybridMBO and addressed here:
#
#  [ABC]        1-op neighborhood exhausted in ~20 steps; scout reset is fully
#               random, discarding all elite knowledge.
#               FIX: Multi-hop neighborhood (1..k genes) + archive-seeded reset.
#
#  [Firefly]    Rank scores cluster when population converges to same accuracy;
#               "brighter" set vanishes -> pure random walk for 64% of gens.
#               FIX: Crowding-distance attraction instead of scalar rank score.
#
#  [PSO]        Velocity collapses to 0 when particle = pbest = gbest;
#               Gaussian mutation sigma -> 0 at end, no escape mechanism.
#               FIX: Velocity noise floor + archive-guided social component.
#
#  [HybridMBO]  FA move with fixed fa_prob produces near-clones when SP1 is
#               converged; no diversity detection.
#               FIX: Adaptive fa_prob that increases on diversity collapse.
#
# Novel contributions (beyond any individual referenced paper):
#   1. Archive-stratified velocity: social component uses the MOST CROWDED
#      archive member (maximally diverse elite guide) rather than gbest,
#      preventing velocity collapse even when the population has converged.
#   2. Diversity-gated perturbation: a lightweight genotype-diversity monitor
#      triggers a targeted archive-seeded crossover burst only when unique
#      genotype ratio drops below a threshold — no wasted budget otherwise.
#   3. Multi-resolution neighborhood sampling: when a food source is about to
#      be abandoned, it escalates from 1-hop to 2-hop to 3-hop neighbors before
#      giving up, exhausting the full local basin before an archive-seeded reset.
#   4. Crowding-distance attraction: firefly movement uses pairwise crowding
#      distance in fitness space (not scalar rank) so attraction persists even
#      after all population members reach the same accuracy level.
# =============================================================================

    """Archive-Guided PSO with Elite Perturbation (APSO-E).

    A population-based multi-objective NAS strategy that surgically addresses
    the convergence bottlenecks of ABC, Firefly, PSO, and HybridMBO.

    Algorithm overview
    ------------------
    Maintains a population of particles and an external non-dominated archive.

    Each generation:
      1. **Velocity update** (PSO core):
            v ← w*v + c1*r1*(pbest - x) + c2*r2*(archive_guide - x)
         where archive_guide is the most-crowded archive member (most diverse
         elite), not the scalar gbest.  This prevents velocity collapse when
         all particles converge to the same genotype.

      2. **Position update**: probability of changing gene d proportional to |v_d|.

      3. **Multi-resolution local search** (ABC-inspired):
         For each particle, try 1-hop → 2-hop → 3-hop neighbors greedily.
         Escalation only happens if the 1-hop search found no improvement,
         replacing ABC's blind abandonment with an exhaustive local basin
         search before escalating.

      4. **Diversity-gated archive crossover** (novel):
         If unique-genotype ratio drops below `div_threshold`, inject
         `n_inject` new individuals created by uniform crossover of two
         random archive members (not random restart). This replaces Firefly's
         genetic fallback and MBO's unchecked cloning.

      5. **Archive update**: standard non-dominated insertion + crowding truncation.

    Parameters
    ----------
    search_space   : SearchSpace
    evaluator      : Evaluator
    pop_size       : int    (default 30)
    max_iterations : int    (default 300)
    archive_size   : int    (default pop_size)
    w              : float  inertia weight (default 0.4)
    c1             : float  cognitive coefficient (default 1.5)
    c2             : float  social coefficient (default 1.5)
    w_decay        : float  multiplicative inertia decay per gen (default 0.99)
    max_hop        : int    max neighborhood hop for local search (default 3)
    div_threshold  : float  diversity ratio triggering archive crossover (default 0.4)
    n_inject       : int    particles injected per diversity event (default 4)
    history        : History | None
    """

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Evaluator,
        pop_size: int = 30,
        max_iterations: int = 300,
        archive_size: int | None = None,
        w: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        w_decay: float = 0.99,
        max_hop: int = 3,
        div_threshold: float = 0.4,
        n_inject: int = 4,
        history: History | None = None,
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.archive_size = archive_size or pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        self.max_hop = max_hop
        self.div_threshold = div_threshold
        self.n_inject = n_inject
        self.history = history or History()

        self._num_ops = search_space.num_ops
        self._num_genes = search_space.num_edges
        self._V_MAX = float(self._num_ops - 1)

        self.evaluations: int = 0
        self.generations: int = 0
        self.population: list[Individual] = []
        self.archive: list[Individual] = []

        # Per-particle state
        self._velocities: list[list[float]] = []
        self._pbests: list[Individual] = []

    # ------------------------------------------------------------------
    # Archive helpers
    # ------------------------------------------------------------------

    def _dominates(self, a: Individual, b: Individual) -> bool:
        return dominates(a, b, self.evaluator.objective_directions)

    def _crowding_distances(self) -> list[float]:
        """Crowding distances of archive members."""
        arch = self.archive
        if len(arch) <= 2:
            return [float("inf")] * len(arch)
        n_obj = len(self.evaluator.objective_directions)
        scores = [0.0] * len(arch)
        for obj in range(n_obj):
            direction = self.evaluator.objective_directions[obj]
            order = sorted(range(len(arch)),
                           key=lambda i: arch[i].fitness[obj] * direction)
            scores[order[0]] = float("inf")
            scores[order[-1]] = float("inf")
            min_v = arch[order[0]].fitness[obj] * direction
            max_v = arch[order[-1]].fitness[obj] * direction
            span = max_v - min_v
            if span == 0:
                continue
            for k in range(1, len(order) - 1):
                idx = order[k]
                if math.isinf(scores[idx]):
                    continue
                prev_v = arch[order[k - 1]].fitness[obj] * direction
                next_v = arch[order[k + 1]].fitness[obj] * direction
                scores[idx] += (next_v - prev_v) / span
        return scores

    def _archive_guide(self) -> Individual:
        """Return the most crowded (most diverse) archive member as guide.

        This is the key innovation over PSO's gbest: instead of the globally
        best scalar individual, we use the one with the highest crowding
        distance — the most isolated elite.  Even when all particles have
        converged to the same high-accuracy architecture, the archive still
        contains diverse members, so the social velocity component remains
        non-zero and particles continue to move.
        """
        if not self.archive:
            return random.choice(self.population)
        cd = self._crowding_distances()
        # With probability 0.7 pick most crowded; with 0.3 pick random archive
        # member (prevents always chasing the same corner of the Pareto front)
        if random.random() < 0.7:
            idx = max(range(len(self.archive)), key=lambda i: cd[i])
        else:
            idx = random.randrange(len(self.archive))
        return self.archive[idx]

    def _archive_update(self, ind: Individual) -> None:
        """Insert ind into archive if non-dominated; truncate by crowding."""
        self.archive = [a for a in self.archive if not self._dominates(ind, a)]
        if not any(self._dominates(a, ind) for a in self.archive):
            if not any(tuple(a.genotype) == tuple(ind.genotype) for a in self.archive):
                self.archive.append(Individual(ind.genotype[:], ind.fitness,
                                               ind.metadata.copy()))
        while len(self.archive) > self.archive_size:
            cd = self._crowding_distances()
            # Remove the most crowded (smallest distance) to maintain spread
            remove_idx = min(range(len(self.archive)), key=lambda i: cd[i])
            del self.archive[remove_idx]

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def _evaluate(self, genotype: list[int]) -> Individual:
        fitness = self.evaluator.evaluate(genotype)
        metadata = {}
        if hasattr(self.search_space, "metadata_from_genotype"):
            metadata = self.search_space.metadata_from_genotype(genotype)
        self.evaluations += 1
        return Individual(genotype[:], fitness, metadata=metadata)

    def _weighted_score(self, ind: Individual) -> float:
        return sum(v * d for v, d in zip(ind.fitness, self.evaluator.objective_directions))

    def _is_better(self, a: Individual, b: Individual) -> bool:
        if a.fitness is None:
            return False
        if b.fitness is None:
            return True
        return self._weighted_score(a) > self._weighted_score(b)

    # ------------------------------------------------------------------
    # Velocity update (PSO core with archive-stratified social component)
    # ------------------------------------------------------------------

    def _update_velocity_and_position(
        self, i: int, guide: Individual, w: float
    ) -> list[int]:
        """Update particle i's velocity and return new genotype.

        Social component uses `guide` (most-crowded archive member) instead
        of gbest.  This keeps social velocity non-zero after convergence.
        A noise floor prevents velocity from reaching exactly 0.
        """
        ind = self.population[i]
        vel = self._velocities[i]
        pbest = self._pbests[i]
        new_geno = list(ind.genotype)

        for d in range(self._num_genes):
            r1 = random.random()
            r2 = random.random()
            v_inertia = w * vel[d]
            v_personal = self.c1 * r1 * (pbest.genotype[d] - ind.genotype[d])
            v_social = self.c2 * r2 * (guide.genotype[d] - ind.genotype[d])
            new_v = v_inertia + v_personal + v_social

            # Noise floor: if velocity is near 0 but particle hasn't converged
            # to the archive guide, inject a small random perturbation.
            # This directly fixes PSO's frozen-particle bottleneck.
            if abs(new_v) < 0.1 and ind.genotype[d] != guide.genotype[d]:
                new_v += random.uniform(-0.3, 0.3)

            vel[d] = max(-self._V_MAX, min(self._V_MAX, new_v))

            # Gene adoption probability proportional to |velocity|
            p_change = min(1.0, abs(vel[d]) / self._V_MAX) if self._V_MAX > 0 else 0.0
            if random.random() < p_change:
                # Bias: if |v_social| > |v_personal|, follow archive guide
                if abs(v_social) >= abs(v_personal):
                    new_geno[d] = guide.genotype[d]
                else:
                    new_geno[d] = pbest.genotype[d]

        self._velocities[i] = vel
        return new_geno

    # ------------------------------------------------------------------
    # Multi-resolution local search (ABC bottleneck fix)
    # ------------------------------------------------------------------

    def _multi_hop_search(self, ind: Individual, max_hop: int) -> Individual | None:
        """Escalating local search: 1-hop → 2-hop → max_hop.

        Unlike ABC's fixed 1-op neighbor, we escalate neighborhood size
        when small hops find nothing.  This exhausts the local basin before
        giving up, avoiding the blind random reset that destroys elite knowledge.

        Returns the best improving neighbor found, or None.
        """
        best = None
        best_score = self._weighted_score(ind)

        for hop in range(1, max_hop + 1):
            # Sample `hop` distinct gene positions to flip
            n_try = min(12, self._num_genes * (self._num_ops - 1))
            for _ in range(n_try):
                positions = random.sample(range(self._num_genes), min(hop, self._num_genes))
                new_geno = ind.genotype[:]
                for pos in positions:
                    choices = [op for op in range(self._num_ops) if op != new_geno[pos]]
                    new_geno[pos] = random.choice(choices)
                candidate = self._evaluate(new_geno)
                s = self._weighted_score(candidate)
                if s > best_score:
                    best = candidate
                    best_score = s
            # Only escalate if no improvement found at this hop level
            if best is not None:
                break

        return best

    # ------------------------------------------------------------------
    # Diversity-gated archive crossover (Firefly / MBO bottleneck fix)
    # ------------------------------------------------------------------

    def _diversity_ratio(self) -> float:
        """Fraction of population with unique genotypes."""
        unique = len(set(tuple(ind.genotype) for ind in self.population))
        return unique / max(1, len(self.population))

    def _archive_crossover_inject(self) -> list[Individual]:
        """Create new individuals by crossing archive members.

        When diversity collapses, this replaces Firefly's random genetic
        fallback with archive-guided crossover: children inherit genes from
        two DIFFERENT archive members, preserving elite knowledge while
        generating genuinely new solutions.
        """
        if len(self.archive) < 2:
            # Fallback: random individual
            return [self._evaluate(self.search_space.random_individual())]
        injected = []
        for _ in range(self.n_inject):
            a, b = random.sample(self.archive, 2)
            child_geno = [
                a.genotype[d] if random.random() < 0.5 else b.genotype[d]
                for d in range(self._num_genes)
            ]
            # Add a small mutation to avoid exact clones of archive members
            for d in range(self._num_genes):
                if random.random() < 0.15:
                    choices = [op for op in range(self._num_ops) if op != child_geno[d]]
                    child_geno[d] = random.choice(choices)
            injected.append(self._evaluate(child_geno))
        return injected

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> list[Individual]:
        """Execute APSO-E and return the final non-dominated archive."""
        K = max(1, self.max_iterations)
        w = self.w
        self.evaluations = 0
        self.generations = 0
        self.archive = []

        # Initialise population
        self.population = []
        self._velocities = []
        self._pbests = []
        for _ in range(self.pop_size):
            geno = self.search_space.random_individual()
            ind = self._evaluate(geno)
            self.population.append(ind)
            self._velocities.append([0.0] * self._num_genes)
            self._pbests.append(Individual(ind.genotype[:], ind.fitness, ind.metadata.copy()))
            self._archive_update(ind)

        self.history.record(
            generation=self.generations,
            evaluations=self.evaluations,
            population=self.population,
            pareto_front=list(self.archive),
        )

        for k in range(1, K + 1):
            guide = self._archive_guide()
            new_population: list[Individual] = []

            # ── Step 1 & 2: Velocity + position update ────────────────
            for i in range(self.pop_size):
                new_geno = self._update_velocity_and_position(i, guide, w)
                ind = self._evaluate(new_geno)
                self._archive_update(ind)

                # Update personal best
                if self._is_better(ind, self._pbests[i]):
                    self._pbests[i] = Individual(ind.genotype[:], ind.fitness,
                                                 ind.metadata.copy())
                new_population.append(ind)

            # ── Step 3: Multi-resolution local search on top half ─────
            # Sort by weighted score; apply local search to better half only
            # (budget-efficient: spend local search on promising particles)
            n_local = max(1, self.pop_size // 2)
            scored = sorted(range(self.pop_size),
                            key=lambda i: self._weighted_score(new_population[i]),
                            reverse=True)
            for i in scored[:n_local]:
                ind = new_population[i]
                improved = self._multi_hop_search(ind, self.max_hop)
                if improved is not None:
                    self._archive_update(improved)
                    if self._is_better(improved, self._pbests[i]):
                        self._pbests[i] = Individual(
                            improved.genotype[:], improved.fitness,
                            improved.metadata.copy())
                    new_population[i] = improved

            # ── Step 4: Diversity-gated archive crossover ─────────────
            div = self._diversity_ratio()
            if div < self.div_threshold and len(self.archive) >= 2:
                injected = self._archive_crossover_inject()
                # Replace the weakest particles
                scored_asc = sorted(range(self.pop_size),
                                    key=lambda i: self._weighted_score(new_population[i]))
                for j, new_ind in enumerate(injected):
                    if j >= len(scored_asc):
                        break
                    self._archive_update(new_ind)
                    replace_idx = scored_asc[j]
                    new_population[replace_idx] = new_ind
                    self._pbests[replace_idx] = Individual(
                        new_ind.genotype[:], new_ind.fitness, new_ind.metadata.copy())
                    # Reset velocity for replaced particle
                    self._velocities[replace_idx] = [0.0] * self._num_genes

            self.population = new_population

            # ── Inertia decay ─────────────────────────────────────────
            w = max(0.1, w * self.w_decay)

            self.generations = k
            self.history.record(
                generation=self.generations,
                evaluations=self.evaluations,
                population=self.population,
                pareto_front=list(self.archive),
            )

        return [Individual(ind.genotype[:], ind.fitness, ind.metadata.copy())
                for ind in self.archive]

class NSGA2SearchStrategy(SearchStrategy):
    """NSGA-II: Non-dominated Sorting Genetic Algorithm II for NAS."""

    def __init__(
        self,
        population: Population,
        selection: Selection,
        crossover: Crossover,
        mutation: Mutation,
        replacement: Replacement,
        evaluator: Evaluator,
        budget: int = 500,
        termination: Termination | None = None,
        history: History | None = None,
        crossover_prob: float = 0.9,
        mutation_prob: float | None = None,
    ):
        from nas_framework.variation import CrossoverMutationVariation
        variation = CrossoverMutationVariation(crossover, mutation)

        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            termination=termination,
            history=history,
            budget=budget,
        )

        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def _assign_rank_and_crowding(self, individuals: list[Individual]) -> list[list[Individual]]:
        from nas_framework.mo_utils import assign_rank_and_crowding
        return assign_rank_and_crowding(individuals, self.evaluator.objective_directions)

    def _select_population(self, combined: list[Individual]) -> list[Individual]:
        from nas_framework.mo_utils import take_pareto_best
        return take_pareto_best(combined, self.population.size, self.evaluator.objective_directions)

    def run(self) -> Population:
        """Execute NSGA-II algorithm (population + offspring elitist selection)."""
        self.population.initialize()
        self._evaluate_offspring(self.population.individuals)

        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            parents = self.selection.select(
                self.population.individuals,
                self.population.size,
                self.evaluator.objective_directions,
            )

            offspring = self.variation.generate(parents, self.population.size)
            self._evaluate_offspring(offspring)

            combined = self.population.individuals + offspring
            # assign rank & crowding on combined pool through helper
            self._assign_rank_and_crowding(combined)
            next_gen = self._select_population(combined)

            self.population.individuals = next_gen
            self.generations += 1
            self._record_history()

        return self.population


class NSGA2EnhancedSearchStrategy(NSGA2SearchStrategy):
    """Budget-safe NSGA-II with an external non-dominated archive.

    This is the NSGA-II variant used for the MO-NAS comparison.  It keeps the
    canonical binary tournament, variation, fast non-dominated sorting and
    crowding-distance survivor selection, while also retaining non-dominated
    solutions found in previous generations.
    """

    def __init__(self, *args, archive_size: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget = int(kwargs.get("budget", 500))
        self.archive_size = archive_size or self.population.size
        self.archive: list[Individual] = []

    @staticmethod
    def _clone(individual: Individual) -> Individual:
        return Individual(
            individual.genotype[:],
            individual.fitness,
            individual.metadata.copy(),
        )

    def _update_archive(self, candidates: list[Individual]) -> None:
        by_genotype: dict[tuple[int, ...], Individual] = {
            tuple(ind.genotype): ind for ind in self.archive
        }
        by_genotype.update({tuple(ind.genotype): ind for ind in candidates})
        archive = compute_pareto_front(
            list(by_genotype.values()),
            self.evaluator.objective_directions,
        )
        if len(archive) > self.archive_size:
            assign_rank_and_crowding(archive, self.evaluator.objective_directions)
            archive.sort(key=lambda ind: ind.crowding_distance, reverse=True)
            archive = archive[: self.archive_size]
        self.archive = [self._clone(ind) for ind in archive]

    def run(self) -> Population:
        """Run NSGA-II without evaluating the initial population twice."""
        self.population.size = min(self.population.size, max(1, self.budget))
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations = 0
        self.archive = []
        self._update_archive(self.population.individuals)
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            remaining = self.budget - self.evaluations
            offspring_count = min(self.population.size, remaining)
            if offspring_count <= 0:
                break

            parents = self.selection.select(
                self.population.individuals,
                self.population.size,
                self.evaluator.objective_directions,
            )
            offspring = self.variation.generate(parents, offspring_count)
            self._evaluate_offspring(offspring)
            combined = self.population.individuals + offspring
            self._assign_rank_and_crowding(combined)
            self.population.individuals = self._select_population(combined)
            self._update_archive(self.population.individuals + offspring)
            self.generations += 1
            self._record_history()

        return [self._clone(ind) for ind in self.archive] if self.archive else self.population


class ABCFireflyStrategy:
    """ABC-Firefly hybrid for multi-objective NAS.

    Structural design
    -----------------
    - ABC skeleton: food sources, employee phase, onlooker phase, scout phase.
    - Employee phase: unchanged 1-op neighbor move (local exploitation within cluster).
    - Onlooker phase: FA multi-gene attraction toward a Pareto-archive member
      instead of 1-op neighbor sampling. This is the inter-cluster jump that
      standard ABC cannot do.
    - Scout phase: objective-stratified reset (K=6 samples, best per extreme:
      min-lat, max-acc, balanced). Guarantees all clusters are seeded.
    - External Pareto archive: maintained every generation, used as FA target
      pool. Never scalar-filtered.

    Parameters
    ----------
    search_space  : CSVSearchSpace
    evaluator     : Evaluator
    pop_size      : int    number of food sources (default 20)
    budget        : int    total evaluation budget
    fa_prob       : float  per-gene FA fire probability in onlooker phase (default 0.5)
    archive_size  : int    max archive size (default pop_size)
    exh_fraction  : float  abandonment_limit = budget / (pop_size * exh_fraction) (default 4.0)
    dmt_fraction  : float  scout suppressed after this fraction of budget (default 0.67)
    history       : History or None
    """

    def __init__(
        self,
        search_space,
        evaluator: Evaluator,
        pop_size: int = 20,
        budget: int = 500,
        fa_prob: float = 0.5,
        archive_size: int | None = None,
        exh_fraction: float = 4.0,
        dmt_fraction: float = 0.67,
        history: History | None = None,
    ):
        self.search_space = search_space
        self.evaluator    = evaluator
        self.pop_size     = pop_size
        self.budget       = budget
        self.fa_prob      = fa_prob
        self.archive_size = archive_size or pop_size
        self.dmt          = int(dmt_fraction * budget)
        self.abandonment_limit = max(3, int(budget / (pop_size * exh_fraction)))
        self.history      = history or History()

        self.evaluations: int = 0
        self.generations: int = 0
        self.food_sources: list  = []   # list[FoodSource]
        self.archive: list       = []   # list[Individual], non-dominated
        self._visited: dict      = {}   # genotype tuple -> fitness tuple

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _eval(self, ind: Individual) -> None:
        """Evaluate ind in-place; use visit cache to avoid duplicate evals."""
        key = tuple(ind.genotype)
        if key in self._visited:
            ind.fitness = self._visited[key]
            if hasattr(self.search_space, "metadata_from_genotype"):
                ind.metadata = self.search_space.metadata_from_genotype(ind.genotype)
            return
        ind.fitness = self.evaluator.evaluate(ind.genotype)
        if hasattr(self.search_space, "metadata_from_genotype"):
            ind.metadata = self.search_space.metadata_from_genotype(ind.genotype)
        self._visited[key] = ind.fitness
        self.evaluations += 1

    def _archive_update(self, ind: Individual) -> None:
        """Add ind to archive if non-dominated; prune dominated members."""
        dirs = self.evaluator.objective_directions
        for a in self.archive:
            if dominates(a, ind, dirs):
                return
        self.archive = [a for a in self.archive if not dominates(ind, a, dirs)]
        # Dedup by genotype
        if not any(tuple(a.genotype) == tuple(ind.genotype) for a in self.archive):
            self.archive.append(Individual(ind.genotype[:], ind.fitness, ind.metadata.copy()))
        # Truncate by crowding distance if over limit
        if len(self.archive) > self.archive_size:
            self._truncate_archive()

    def _truncate_archive(self) -> None:
        """Remove the archive member with smallest crowding distance."""
        if len(self.archive) <= self.archive_size:
            return
        dirs = self.evaluator.objective_directions
        n_obj = len(dirs)
        scores = [0.0] * len(self.archive)
        for obj in range(n_obj):
            order = sorted(range(len(self.archive)),
                           key=lambda i: self.archive[i].fitness[obj] * dirs[obj])
            scores[order[0]]  = float("inf")
            scores[order[-1]] = float("inf")
            f_min = self.archive[order[0]].fitness[obj]  * dirs[obj]
            f_max = self.archive[order[-1]].fitness[obj] * dirs[obj]
            if f_max == f_min:
                continue
            for k in range(1, len(order) - 1):
                idx = order[k]
                if math.isinf(scores[idx]):
                    continue
                prev_v = self.archive[order[k-1]].fitness[obj] * dirs[obj]
                next_v = self.archive[order[k+1]].fitness[obj] * dirs[obj]
                scores[idx] += (next_v - prev_v) / (f_max - f_min)
        remove = min(range(len(self.archive)), key=lambda i: scores[i])
        del self.archive[remove]

    def _neighbor_1op(self, ind: Individual) -> Individual:
        """Standard ABC 1-op neighbor: change one random gene."""
        geno = ind.genotype[:]
        idx  = random.randint(0, len(geno) - 1)
        choices = [op for op in range(self.search_space.num_ops) if op != geno[idx]]
        geno[idx] = random.choice(choices)
        return Individual(geno)

    def _fa_move(self, ind: Individual, target: Individual) -> Individual:
        """FA multi-gene attraction: adopt target gene with prob fa_prob per gene.

        Unlike 1-op neighbor, this can change multiple genes in one step,
        bridging the Hamming gap between distant clusters.
        """
        geno = ind.genotype[:]
        for i in range(len(geno)):
            if geno[i] != target.genotype[i] and random.random() < self.fa_prob:
                geno[i] = target.genotype[i]
        return Individual(geno)

    def _pareto_accept(self, fs, candidate: Individual) -> bool:
        """Accept candidate into food source using Pareto dominance.

        - candidate dominates current: always accept.
        - current dominates candidate: always reject.
        - non-dominated pair: accept with 50% probability (diversity preservation).
        """
        dirs = self.evaluator.objective_directions
        curr = fs.individual
        if dominates(candidate, curr, dirs):
            fs.individual  = candidate
            fs.trial_count = 0
            return True
        if dominates(curr, candidate, dirs):
            fs.trial_count += 1
            return False
        # Non-dominated: accept with 50% to maintain diversity
        if random.random() < 0.5:
            fs.individual  = candidate
            fs.trial_count = 0
            return True
        fs.trial_count += 1
        return False

    def _stratified_reset(self) -> Individual:
        """Sample K=6 random archs; return best on a random extreme objective."""
        K    = 6
        dirs = self.evaluator.objective_directions
        candidates = []
        for _ in range(K):
            ind = Individual(self.search_space.random_individual())
            self._eval(ind)
            if self.termination_check():
                break
            candidates.append(ind)
        if not candidates:
            ind = Individual(self.search_space.random_individual())
            self._eval(ind)
            return ind
        extreme = random.randint(0, 2)
        if extreme == 0:   # min latency
            return max(candidates, key=lambda x: x.fitness[1] * dirs[1])
        elif extreme == 1:  # max accuracy
            return max(candidates, key=lambda x: x.fitness[0] * dirs[0])
        else:              # balanced: non-dominated among K
            front = compute_pareto_front(candidates, dirs)
            return random.choice(front) if front else candidates[0]

    def termination_check(self) -> bool:
        limit = getattr(self, '_budget_main', self.budget)
        return self.evaluations >= limit

    def _record(self) -> None:
        inds = [fs.individual for fs in self.food_sources]
        self.history.record(
            generation=self.generations,
            evaluations=self.evaluations,
            population=inds,
            pareto_front=list(self.archive),
        )

    # ------------------------------------------------------------------
    # ABC phases
    # ------------------------------------------------------------------

    def _employee_phase(self) -> None:
        """1-op neighbor move per food source; Pareto-aware acceptance."""
        for fs in self.food_sources:
            if self.termination_check():
                return
            neighbor = self._neighbor_1op(fs.individual)
            self._eval(neighbor)
            self._archive_update(neighbor)
            self._pareto_accept(fs, neighbor)

    def _onlooker_phase(self) -> None:
        if not self.archive:
            return
        fixed_lat = self._is_fixed_latency()
        dirs = self.evaluator.objective_directions
        n_arch = len(self.archive)

        if not fixed_lat and n_arch >= 2:
            scores = [0.0] * n_arch
            for obj in range(len(dirs)):
                order = sorted(range(n_arch),
                               key=lambda i: self.archive[i].fitness[obj] * dirs[obj])
                scores[order[0]] = float("inf")
                scores[order[-1]] = float("inf")
                f_min = self.archive[order[0]].fitness[obj] * dirs[obj]
                f_max = self.archive[order[-1]].fitness[obj] * dirs[obj]
                if f_max == f_min:
                    continue
                for k in range(1, len(order) - 1):
                    idx = order[k]
                    if math.isinf(scores[idx]):
                        continue
                    prev_v = self.archive[order[k-1]].fitness[obj] * dirs[obj]
                    next_v = self.archive[order[k+1]].fitness[obj] * dirs[obj]
                    scores[idx] += (next_v - prev_v) / (f_max - f_min)
            weights = [10.0 if math.isinf(s) else max(1e-3, s) for s in scores]
        else:
            weights = None

        last_arch_size = len(self.archive)
        for fs in self.food_sources:
            if self.termination_check():
                return
            if fixed_lat or weights is None:
                neighbor = self._neighbor_1op(fs.individual)
                self._eval(neighbor)
                self._archive_update(neighbor)
                self._pareto_accept(fs, neighbor)
            else:
                if len(self.archive) != last_arch_size:
                    n_arch = len(self.archive)
                    scores = [0.0] * n_arch
                    for obj in range(len(dirs)):
                        order = sorted(range(n_arch),
                                       key=lambda i: self.archive[i].fitness[obj] * dirs[obj])
                        scores[order[0]] = float("inf")
                        scores[order[-1]] = float("inf")
                        f_min = self.archive[order[0]].fitness[obj] * dirs[obj]
                        f_max = self.archive[order[-1]].fitness[obj] * dirs[obj]
                        if f_max == f_min:
                            continue
                        for k in range(1, len(order) - 1):
                            idx = order[k]
                            if math.isinf(scores[idx]):
                                continue
                            prev_v = self.archive[order[k-1]].fitness[obj] * dirs[obj]
                            next_v = self.archive[order[k+1]].fitness[obj] * dirs[obj]
                            scores[idx] += (next_v - prev_v) / (f_max - f_min)
                    weights = [10.0 if math.isinf(s) else max(1e-3, s) for s in scores]
                    last_arch_size = n_arch
                target = random.choices(self.archive, weights=weights, k=1)[0]
                moved = self._fa_move(fs.individual, target)
                self._eval(moved)
                self._archive_update(moved)
                self._pareto_accept(fs, moved)

    def _scout_phase(self) -> None:
        """Stratified reset for exhausted food sources (early search only)."""
        if self.evaluations >= self.dmt:
            return
        for fs in self.food_sources:
            if fs.trial_count >= self.abandonment_limit:
                if self.termination_check():
                    return
                fresh = self._stratified_reset()
                self._archive_update(fresh)
                fs.reset(fresh)
    def _is_fixed_latency(self) -> bool:
        if len(self.archive) < 2:
            return False
        lats = [a.fitness[1] for a in self.archive if a.fitness]
        return max(lats) - min(lats) < 1e-6

    def _endpoint_local_search(self, extra: int) -> None:
        dirs = self.evaluator.objective_directions
        if not self.archive:
            return
        hard_limit = self.evaluations + extra
        i = 0
        attempts = 0
        max_attempts = extra * 10
        while self.evaluations < hard_limit and attempts < max_attempts:
            # Refresh extremes each iteration so they track archive updates
            lat_extreme = min(self.archive, key=lambda a: a.fitness[1] * dirs[1])
            acc_extreme = max(self.archive, key=lambda a: a.fitness[0] * dirs[0])
            # Only use lat extreme if latency spread is meaningful (>1e-4)
            lats = [a.fitness[1] for a in self.archive]
            use_lat = (max(lats) - min(lats)) > 1e-4
            if use_lat:
                seed = lat_extreme if i % 2 == 0 else acc_extreme
            else:
                seed = acc_extreme
            neighbor = self._neighbor_1op(seed)
            self._eval(neighbor)
            self._archive_update(neighbor)
            i += 1
            attempts += 1

    def run(self) -> list[Individual]:
        from nas_framework.population import FoodSource
        self.food_sources = []
        self.archive      = []
        self._visited     = {}
        self.evaluations  = 0
        self.generations  = 0

        # Reserve 5% budget for endpoint polish
        extra        = max(1, int(0.05 * self.budget))
        self._budget_main = self.budget - extra   # main loop stops here

        for _ in range(self.pop_size):
            geno = self.search_space.random_individual()
            ind  = Individual(geno)
            self._eval(ind)
            self.food_sources.append(FoodSource(ind))
            self._archive_update(ind)

        self._record()

        while not self.termination_check():
            self._employee_phase()
            if self.termination_check():
                break
            self._onlooker_phase()
            if self.termination_check():
                break
            self._scout_phase()
            self.generations += 1
            self._record()

        self._endpoint_local_search(extra)

        return [Individual(a.genotype[:], a.fitness, a.metadata.copy())
                for a in self.archive]


