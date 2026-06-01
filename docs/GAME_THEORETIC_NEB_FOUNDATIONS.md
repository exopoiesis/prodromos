# Game-Theoretic Foundations of NEB:
## Deep analogies from optimization theory, distributed systems, and multi-agent games

**Контекст:** проект Third Matter, проблема same-basin trap в FeS V_Fe NEB
**Cross-ref:** `ALTERNATIVES_AND_ROLE_AWARE_NEB.md` (landscape + lit-scan)
**Статус:** концептуальный framework + concrete новая формулировка NEB-AGM. Research direction.

---

## Часть 0. Контекст и мотивация

Исходная идея: **каждый image NEB — ячейка с локальной моделью PES, накапливающая знания**. Стандартный NEB трактует все 9 images как однородные degrees of freedom одной optimization variable. Мы хотим heterogeneity: каждый image = independent agent с local surrogate.

В этом документе — глубокий поиск аналогий в:
- Variational analysis (ADMM, Domain Decomposition)
- Topology (Discrete Morse, Persistent Homology)
- Probability (EnKF, Replica Exchange, Wasserstein flows)
- Optimal control (DDP, MPC)
- ML (Mixture of Experts, GNN, Continual Learning)
- **Game theory** (Potential Games, Stackelberg, Mean-Field Games, Shapley, No-Regret)

И конкретная новая формулировка **NEB-AGM (NEB as Adaptive Game with Memory)** — гибрид, объединяющий productive analogies.

**Главный тезис:** same-basin trap — это **equilibrium selection problem**, не technical bug. Это переопределение открывает арсенал теории игр для атаки на проблему.

---

## Часть 1. Variational viewpoint: ADMM + Domain Decomposition ⭐

### 1.1. ADMM (Alternating Direction Method of Multipliers)

**Boyd et al. 2011** — стандартный метод distributed convex optimization. Раскладывает:
```
minimize    Σᵢ fᵢ(xᵢ)
subject to  A x = b   (coupling constraint)
```
через **augmented Lagrangian** с dual variables λ:
```
L_ρ(x, λ) = Σᵢ fᵢ(xᵢ) + λᵀ(Ax − b) + (ρ/2)‖Ax − b‖²
```
и iterative update:
```
xᵢ^{k+1} = argmin_x  fᵢ(x) + (ρ/2)‖A_i x + ... − b + λᵏ/ρ‖²     [local]
λ^{k+1}   = λᵏ + ρ(Ax^{k+1} − b)                                  [dual update]
```

### 1.2. NEB переформулирован как ADMM

**Стандартное NEB-уравнение силы:**
```
F_i = −∇V(x_i)|_⊥ + k_spring (Δs_right − Δs_left) τ̂_i
```

**Эквивалентная ADMM-форма:**
- **Local objectives** f_i(x_i) = V(x_i)  (можно заменить на surrogate V̂_i!)
- **Coupling constraint** |x_{i+1} − x_i| = h_target  (equidistant)
- **Dual variables** λ_i  = **spring tension** между i и i+1
- **Augmented Lagrangian:**
  ```
  L_ρ = Σᵢ V(xᵢ) + Σᵢ λᵢ (|xᵢ₊₁−xᵢ| − h) + (ρ/2) Σᵢ (|xᵢ₊₁−xᵢ| − h)²
  ```

**Что это даёт нового:**
1. **Per-image local model:** swap V(x_i) → V̂_i(x_i) (локальный GP/quadratic) тривиально, ADMM не меняется
2. **Provable convergence:** для convex local f есть theorems (Boyd 2011, He & Yuan 2012). Для non-convex (наш случай) — recent extensions (Wang et al. 2019, "ADMM for nonconvex problems")
3. **Parallelism:** x_i updates по чётным/нечётным images независимы → 2x speedup
4. **Natural handling of stuck images:** если local minimization не сходится → λ накапливается → ADMM "форсирует" движение через dual penalty

### 1.3. Domain Decomposition Methods (DDM)

Numerical PDE community использует **Schwarz alternating, FETI, BDDC** методы 40 лет. Каждый subdomain решает PDE локально, interface conditions сглаживают.

**Прямая аналогия:**
- Subdomain Ωᵢ ↔ neighborhood image i на PES
- Local solver ↔ local minimization V̂_i
- Interface condition ↔ spring constraint
- **Атлас многообразия** = набор перекрывающихся карт = images. Это не метафора, это формально совпадает с manifold theory.

**Mature theory:** Toselli & Widlund "Domain Decomposition Methods" (2004), 500 страниц rigorous math. Готовый к импорту.

### 1.4. Multigrid view

**Multigrid methods** используют hierarchy of resolutions. В NEB:
- Coarse path: 5 images (быстрая convergence на large-scale topology)
- Fine path: 9 images (точная локализация saddle)
- Cycle: V-cycle / W-cycle между уровнями

**Не публиковалось в NEB.** Стандартная практика — фиксированное N=9 images.

---

## Часть 2. Topological viewpoint: Discrete Morse + Persistent Homology

### 2.1. Discrete Morse Theory (Forman 2002)

Combinatorial flows на simplicial complex K. Discrete Morse function f: K → ℝ с критическими симплексами (нет pairing с соседями). Аналог классической Morse theory, но дискретный.

**Применение к NEB:**
- Каждый image живёт на симплексе (vertex/edge/face/...)
- Локальная f_i на каждом симплексе строится из локальной модели V̂_i
- Combinatorial gradient flow = discrete analog NEB updates
- **Преимущество:** explicit combinatorial structure → algorithmic guarantees

### 2.2. Persistent Homology (PH) для NEB diagnostics

**Edelsbrunner et al. 2002.** Tracking topological features (connected components, loops, voids) как параметр меняется.

**Конкретный диагностический инструмент для same-basin trap:**

1. Для каждого image i вычислить **persistence diagram** локальной PES V̂_i (через sublevel filtration)
2. Persistence diagram содержит **barcode** — длительность каждой топологической feature
3. **Basin signature:** 0-dim persistence долго живёт (deep minimum)
4. **Saddle signature:** 1-dim persistence появляется (handle)
5. **Cross-image comparison:** Wasserstein/bottleneck distance между barcodes

**Диагностика same-basin trap:**
```
if bottleneck_distance(PH(V̂_1), PH(V̂_5)) < threshold:
    → images 1 и 5 имеют топологически идентичную local environment
    → они в одной basin
    → ALERT: same-basin trap detected
```

**Это formal automated detector нашего hand-crafted "structural sanity gate".**

**Lit-status:** не нашёл ни одной NEB-работы с PH (lit-scan part 6.5). Применения PH в materials есть (Hiraoka et al. 2016, "Hierarchical structures in amorphous solids"), но не для transition paths.

### 2.3. Connection с mountain pass theorem

Mountain pass theorem гарантирует существование saddle. Persistent homology **конструктивно строит** invariant класс гомологий, отвечающий за saddle. Это может дать **алгоритмический mountain pass theorem** — explicit construction вместо variational existence proof.

---

## Часть 3. Probabilistic viewpoint: EnKF + Replica Exchange + Wasserstein flows

### 3.1. Ensemble Kalman Filter (EnKF)

**Evensen 1994.** Data assimilation: каждый ensemble member имеет state estimate, updates через ensemble covariance.

**EnKF для NEB:**
- Каждый image i = ensemble member со state x_i и local covariance Σ_i
- Local model V̂_i обновляется EnKF-style на каждой DFT evaluation
- **Cross-image consensus** через ensemble covariance структуру

EnKF используется в meteorology с **тысячами** локальных моделей в реальном времени. Технология готова, надо только адаптировать notation.

### 3.2. Replica Exchange / Parallel Tempering

**Sugita & Okamoto 1999.** Replicas at different temperatures обмениваются конфигурациями через Metropolis.

**Связь с NEB:** в стандартном NEB images **не обмениваются** — только через springs. Если каждый image "температурный" replica (с своим T_i), они могут swap конфигурациями. Это даёт:
- **Stuck escape:** замёрзший image обменивается с горячим, prikol получает kick
- **Heterogeneous exploration:** разные images на разных temperatures одновременно

**Combined approach: Replica Exchange NEB (REN-NEB)** — упоминался в нескольких papers 2010-х (например, Kim et al. 2014), но не mainstream.

### 3.3. Wasserstein gradient flows

**Jordan-Kinderlehrer-Otto (JKO) scheme 1998.** Gradient flow on probability distributions in Wasserstein-2 metric.

**Глубокая связь с MEP:**
- Каждый image параметризует distribution p_i (не точку!)
- Path = flow в Wasserstein space W₂
- MEP = geodesic в этом метрическом пространстве

**Это где optimal transport встречается с MEP**, и обычно опускается в NEB-литературе. Связь через **Brenier-Benamou formula** (Brenier 1991, Benamou 2000) даёт fluid-dynamic interpretation — path как поток "массы вероятности" из A в B с минимальной кинетической энергией.

**Practical implication:** вместо point images можно использовать **distribution images** (parameterized GMMs or normalizing flows). Это объединяется с TPT (committor function) и Boltzmann generators в общий framework.

---

## Часть 4. Optimal Control viewpoint: DDP + MPC ⭐

### 4.1. Differential Dynamic Programming (DDP)

**Jacobson & Mayne 1970.** Trajectory optimization через **локальные quadratic approximations** value function на каждой stage, stitched via Bellman recursion.

**Прямой импорт в NEB:**

**Forward pass (накопление информации):**
```
For each image i (forward):
  V̂_i(x) = V(x_i⁰) + g_iᵀ(x − x_i⁰) + ½(x − x_i⁰)ᵀ H_i (x − x_i⁰)
  # g_i — gradient at x_i⁰ (cheap from current DFT)
  # H_i — local Hessian estimate (BFGS-style update per image)
```

**Backward pass (consensus update):**
```
For each image i (backward from N to 1):
  K_i = compute_gain_matrix(H_i, H_{i+1}, spring_coupling)
  # Bellman: value-to-go = local cost + propagated future cost
  
  next_step_direction_i = −K_i · g_i  # incorporates downstream info
```

**Per-image quadratic model V̂_i** — это буквально то, что DDP делает на каждой stage. **Каждый image держит свой Hessian**, обновляемый BFGS-style из локальных DFT. Сейчас в стандартном NEB — один глобальный BFGS на конкатенированный state vector.

**Mature в robotics:** Tassa, Mansard, Todorov (iLQR/DDP), используется в MPC для humanoid control с >100 DOF в реальном времени.

**Lit-status:** **не публиковалось в NEB context** (проверено).

### 4.2. Model Predictive Control (MPC) на chain graph

MPC: rolling horizon optimization с local model.
- Image i имеет MPC controller с horizon H
- Looks ahead H steps along path
- Local model V̂_i обновляется каждую iteration
- **Distributed MPC** (Camponogara et al. 2002) — каждый MPC консультируется с соседями

Это даёт **predictive component** — каждый image "предвидит" что будут делать соседи.

### 4.3. Связь с Pontryagin Maximum Principle

PMP: optimal trajectory γ*(s) удовлетворяет Hamiltonian equations:
```
γ̇ = ∂H/∂p
ṗ = −∂H/∂γ
```
где p — costate (Lagrange multiplier).

**В NEB:** spring tension λ_i ↔ discrete costate p_i. ADMM-формулировка (Часть 1) — это разностный аналог PMP.

---

## Часть 5. ML viewpoint: MoE + GNN + Continual Learning

### 5.1. Mixture of Experts (MoE) — ровно Lance/MaPE

**Shazeer et al. 2017** (sparse MoE) → Lance 2026.

**Для NEB:**
- Каждый image = эксперт по region of PES
- Gating function g(x) = role assignment (MaPE-аналог)
- Sparse activation: только relevant experts активны при evaluation x
- Training: experts специализируются автоматически (load balancing loss)

**Имплементация:**
- 9 lightweight neural networks (по числу images)
- 1 gating network — выбирает который эксперт активен
- Gating обучается end-to-end через differentiable assignment

### 5.2. Graph Neural Networks (GNN)

Images = nodes, springs = edges. Message passing:
```
m_{i→j} = MLP_message(x_i, x_j, edge_features)
x_j^{k+1} = MLP_update(x_j^k, Σ_i m_{i→j})
```

**Replacement для fixed springs:** spring constants k_spring заменяются learned attention weights между images. Каждый image учит "с кем coordinated сильнее".

**Recent precedent:** Equivariant GNNs (Satorras et al. 2021, Schütt et al. 2021, NequIP/Allegro/MACE) уже используются для **PES** в MLIP. Расширение на **path-level** GNN, где nodes = images path, — естественное.

### 5.3. Continual Learning

Каждый image движется → его локальная модель должна адаптироваться без забывания накопленного.

**Применимые методы:**
- **Elastic Weight Consolidation (Kirkpatrick et al. 2017):** Fisher information-weighted regularization
- **Experience Replay:** keep buffer of past (x, ∇V) pairs, retrain periodically
- **Progressive Networks:** добавлять capacity вместо переписывать

Это решает проблему "image движется на 10 Å за optimization, его local GP from initial position бесполезен в final position".

---

## Часть 6. Game Theory viewpoint ⭐⭐⭐ (самый плодородный)

### 6.1. NEB как Potential Game

**Monderer & Shapley 1996.** Potential game: игра, где Nash equilibrium совпадает с локальным минимумом scalar potential function.

**Формальная NEB-as-game:**
- Игроки: N = {1, ..., 9} (images)
- Strategy игрока i: позиция x_i ∈ ℝᵈ
- Payoff игрока i:
  ```
  u_i(x_1, ..., x_N) = −V(x_i) − (k/2)[(x_i − x_{i−1})² + (x_{i+1} − x_i)²]
  ```
- Potential function:
  ```
  Φ(x_1, ..., x_N) = Σᵢ V(x_i) + (k/2) Σᵢ (x_{i+1} − x_i)²
  ```

**Свойство potential game:** ∂u_i/∂x_i = −∂Φ/∂x_i для всех i. **Проверка:** ∂u_i/∂x_i = −∇V(x_i) − k[(x_i−x_{i−1}) − (x_{i+1}−x_i)]. ∂Φ/∂x_i = ∇V(x_i) + k[(x_i−x_{i−1}) − (x_{i+1}−x_i)]. ✓

→ **NEB — формально potential game.**

**Что это даёт:**

1. **Existence:** Monderer-Shapley theorem гарантирует существование pure Nash equilibrium → independent proof существования NEB solution (parallel к mountain pass theorem)

2. **Convergence:** **best-response dynamics** в potential games сходятся к Nash equilibrium монотонно (Φ убывает). Это объясняет почему наивный gradient descent на images работает.

3. **Multiplicity = MEP multiplicity:**
   Potential games могут иметь множество Nash equilibria. **Каждое equilibrium = candidate path.**
   - "Хорошее" equilibrium = true MEP (correct saddle)
   - "Плохое" equilibrium = same-basin trap, alternate saddle, etc.
   
   **Переопределение проблемы:** same-basin trap — это не technical failure, это convergence к **wrong Nash equilibrium**. Это **equilibrium selection problem**, которая имеет богатую теорию (Harsanyi-Selten 1988 "A General Theory of Equilibrium Selection", risk dominance, evolutionary stability).

4. **Mechanism design:** мы проектируем payoff (force law) такой, чтобы selfish play сходился к глобально-оптимальному equilibrium. **Текущий NEB — плохо спроектированный mechanism**: payoff структура допускает collapse к плохим equilibria.

   **Better mechanism design ideas:**
   - Asymmetric springs: разная k для basin vs transition regions
   - Time-varying payoff: ranged k(t) с annealing
   - Side payments: extra reward для images, escaping local minima
   - Punishment terms: penalty за collapse в neighbor basin

### 6.2. CI-NEB как Stackelberg Game

**Stackelberg 1934.** Игра с commitment: leader двигается первым, followers оптимально отвечают.

**CI-NEB структурно — Stackelberg:**
- Climbing image (leader) commits to climbing direction
- Все остальные images (followers) реагируют, поддерживая path topology

**Stackelberg equilibrium** обычно **лучше Nash** для leader. Это формально объясняет почему CI-NEB лучше plain NEB — leader-follower structure breaks degenerate Nash equilibria.

**Extensions:**
- **Multi-leader Stackelberg:** несколько images leader на разных стадиях
- **Dynamic Stackelberg:** leadership переходит между images по мере evolution path
- **Stackelberg-Nash hybrid:** leaders играют Stackelberg между собой, followers Nash

**Это даёт принципиально новые алгоритмы**, без аналога в NEB-литературе.

### 6.3. Mean-Field Games (MFG)

**Lasry & Lions 2006-2007, Huang-Caines-Malhamé.** Continuum limit многоагентной игры. Каждый агент отвечает на average field остальных через **self-consistent HJB + Fokker-Planck pair**:

```
−∂V/∂t + H(x, ∇V) = F[m]        (HJB: each agent's value function)
∂m/∂t − div(m ∇_p H) = 0          (FP: distribution evolves)
m(0) = m_0, m(T) = m_T            (boundary distributions)
```

**Для NEB при N→∞:**
- Path γ(s) → flow of agents
- Distribution m(s, x) = density of images at arclength s
- Self-consistent equations связывают local model agent (HJB) с global field (FP)

**Глубокая связь с optimal transport:**
- MFG в Lagrangian форме = Benamou-Brenier formula
- Benamou-Brenier connects to Wasserstein gradient flow (Часть 3.3)
- Wasserstein flow connects to TPT через committor

**Это треугольник** Game-Theory ↔ OT ↔ TPT, **который никто не замкнул в NEB-литературе**.

**Practical:** даёт **continuous-image NEB** где N→∞ становится принципом, а не aппроксимацией.

### 6.4. Cooperative Game Theory: Shapley Value

**Shapley 1953.** Справедливое распределение payoff кооперативной игры:
```
φ_i = Σ_{S ⊆ N\{i}} (|S|! (n−|S|−1)!)/n! · [v(S∪{i}) − v(S)]
```
где v(S) — worth coalition S.

**Применение к NEB:**
- v(S) = "сколько барьер reduction обеспечивает coalition S images"
- φ_i = маржинальный вклад image i, усреднённый по всем permutations
- **Вычисление:** Monte Carlo по random permutations (Castro et al. 2009)

**Использование:**

1. **Adaptive DFT budget allocation:** images с высокой Shapley → больше DFT compute. Active learning natural.

2. **Image pruning:** φ_i ≈ 0 → image redundant. Можно убрать из path. Variable-N NEB.

3. **Stuck detection:** φ_i резко падает между iterations → image застрял, не приносит пользу. ALERT.

4. **Insurance для convergence:** Aumann-Shapley value даёт **continuous extension** — pricing для contribution каждого image в continuous limit.

**Lit-status:** **ни одной NEB-работы с Shapley** (verified lit-scan). Это открытое направление.

### 6.5. No-Regret Learning + Correlated Equilibrium

**Online learning** framing: каждый image — agent с no-regret algorithm.

**Algorithms:**
- **Online Gradient Descent (OGD):** Zinkevich 2003
- **Follow The Regularized Leader (FTRL):** Hazan 2016 textbook
- **Multiplicative Weights / Hedge:** Freund-Schapire 1997
- **EXP3 / EXP3.P:** для bandit feedback (когда gradient unknown)

**Theorem (Foster-Vohra 1997, Hart-Mas-Colell 2000):** если все агенты используют no-regret algorithm, joint play сходится к **correlated equilibrium** (Aumann 1974). Это weakly slabier чем Nash, но достижимо без full information и без best-response computation.

**Для federated NEB:** каждый image не знает global PES, только local через DFT calls. No-regret learning гарантирует convergence к correlated equilibrium without explicit coordination. **Это формальный federated framework.**

**Practical algorithm:**
```
For each image i, each iteration:
  Observe local gradient g_i (cheap, from V̂_i)
  Update local model V̂_i (continual learning)
  Take FTRL/OGD step: x_i^{k+1} = x_i^k − η·g_i + regularization
  Exchange spring tension λ with neighbors (communication)
```

**Что нового:** rigorous convergence гарантии **под минимальной информацией** на image. Не нужно глобальной convexity, smoothness, etc.

### 6.6. Evolutionary Game Theory (EGT)

**Maynard Smith 1973.** Strategies эволюционируют по replicator dynamics:
```
ẋ_i = x_i (u_i(x) − ū(x))
```

**Для NEB:** **различные path candidates** конкурируют. Selection pressure favors paths with lower barrier. Это population-based search (как matrix PSO из lit-scan), но с **rigorous evolutionary theorem** (Folk theorem для replicator dynamics).

**ESS (Evolutionarily Stable Strategy):** path который не может быть invaded мутациями. Это **stronger than Nash** — гарантирует robust convergence.

---

## Часть 7. Синтез: предлагаемая формулировка NEB-AGM

**NEB-AGM = NEB as Adaptive Game with Memory.**

Гибрид, объединяющий productive analogies:

### 7.1. Architecture

```
For each image i ∈ {1, ..., N}:
  • Local quadratic model V̂_i(x) (DDP-style)
    - Initialized: V(x_i⁰) + 0·(x − x_i⁰) + λ·I  (trivial Hessian)
    - Updated: BFGS-per-image на (x_i, ∇V(x_i)) history
    
  • Persistence diagram PH_i (топологическая signature local PES)
    - Computed periodically via sublevel filtration V̂_i
    
  • Role label r_i ∈ {basin, ridge, saddle, climber, stuck, transition}
    - Assigned by classifier C(eigenvals(H_i), |g_i|, displacement_var, PH_i)
    
  • Shapley value φ_i (importance score)
    - Estimated via Monte Carlo permutations every K iterations
    
  • No-regret state (FTRL accumulator) ω_i
    - For role-conditioned update step
```

### 7.2. Game structure

- **Players:** N images
- **Strategies:** positions x_i
- **Payoff (role-conditioned):**
  ```
  u_i(x; r_i, λ) = −V̂_i(x_i) − Σ_{j∈neighbors} λ_{ij}(|x_i − x_j| − h_target)
                   + role_bonus(r_i, x_i)
  ```
- **Mechanism design:** role_bonus designed to break degenerate Nash equilibria. Examples:
  - basin role: small bonus for moving toward path interior
  - stuck role: large bonus for any move away from current basin
  - climber role: bonus for moving along τ̂ (ascending)

### 7.3. Update rule (per iteration)

```
1. Observation phase:
   For each image i:
     If φ_i high OR confidence_low(V̂_i):
       Evaluate true V(x_i), ∇V(x_i) via DFT
       Update V̂_i (continual learning)
     Else:
       Use surrogate V̂_i

2. Role assignment phase:
   For each i: r_i ← C(eigenvals(H_i), |g_i|, PH_i, neighbor_dist)

3. Game step (ADMM-style):
   For each i (parallel via even/odd splitting):
     x_i^{k+1} = argmin_x [V̂_i(x) − u_i(x; r_i, λ^k) + (ρ/2)||consensus residual||²]
   
4. Dual update:
   For each spring (i, i+1):
     λ_{i,i+1}^{k+1} = λ_{i,i+1}^k + ρ(|x_{i+1}^{k+1} − x_i^{k+1}| − h_target)

5. Diagnostic phase (every K iterations):
   - Compute Shapley {φ_i}
   - Compute pairwise PH bottleneck distances → same-basin detection
   - Mechanism design adjustment: if same-basin detected → modify role bonuses

6. Convergence check:
   - max|F_i| < fmax  (standard)
   - AND  Shapley distribution stable
   - AND  no role transitions in last K iterations
   - AND  PH-based diagnostics PASS (no same-basin signature)
```

### 7.4. Theoretical guarantees (что можем доказать)

| Property | Source | Conditions |
|---|---|---|
| Existence Nash equilibrium | Potential game (§6.1) | Φ coercive |
| Convergence к Nash | Best-response in potential game (§6.1) | Lipschitz gradients |
| Provable convergence ADMM step | Boyd 2011 / Wang 2019 | Either convex local f or specific non-convex conditions |
| No-regret bound | Hazan 2016 | Convex local f̂ |
| Shapley axiom satisfaction | Shapley 1953 | Always |

**Honest:** не все одновременно для non-convex V. Открытая теоретическая проблема — **consistency of all guarantees** в нашем setting. Это часть novelty для methodology paper.

### 7.5. Empirical predictions

1. **Same-basin trap detection** через PH bottleneck distance → automated termination + restart suggestion. Должно поймать наши известные mackinawite / pentlandite / marcasite same-basin incidents retrospectively.

2. **Compute efficiency:** per-image surrogates V̂_i + Shapley-guided DFT allocation должны дать **2-5× reduction в DFT calls** (similar к ML-NEB, but per-image localization tighter).

3. **Robust convergence на multi-saddle systems:** mechanism design + role differentiation должны меньше залипать на дёгенеративных Nash equilibria.

---

## Часть 8. Сравнительная таблица всех angles

| Angle | Что даёт уникального | Зрелость теории | Реализация difficulty | Связь с our same-basin problem |
|---|---|---|---|---|
| **ADMM / DDM** | Provable convergence + parallelism + per-image surrogate trivially | Mature | Medium | Indirect (через λ dual) |
| **Persistent Homology** | Automated same-basin detection | Mature | Medium-High (PH libs) | **Direct diagnostic** |
| **EnKF** | Bayesian per-image model with consensus | Mature | Low (libraries exist) | Indirect |
| **Replica Exchange** | Stuck escape via temperature swaps | Mature | Low | **Direct treatment** |
| **Wasserstein flows** | Distribution-valued images, OT connection | Maturing | High | Indirect |
| **DDP** | Per-image quadratic V̂_i с rigorous backward pass | Mature in robotics | Medium | Indirect |
| **MoE / Lance** | Role-aware via gating function | Mature | Low (MoE libs) | **Direct (MaPE analog)** |
| **GNN** | Learned attention vs fixed springs | Mature | Low | Indirect |
| **Potential Game** | Existence + multiplicity + **equilibrium selection framing** | Mature | Conceptual | **Reframes the problem** |
| **Stackelberg / CI-NEB** | Leader-follower mechanism, explains CI-NEB | Mature | Low | Indirect |
| **Mean-Field Game** | Continuous-image limit, OT-TPT connection | Mature in math | High | Indirect |
| **Shapley value** | Importance scoring per image | Mature | Medium (MC estimation) | **Diagnostic** |
| **No-Regret learning** | Federated convergence guarantee minimal info | Mature | Medium | Indirect |
| **Evolutionary game theory** | ESS robust to mutations | Mature | Low | Indirect |

**Топ-3 по "direct relevance to our same-basin pain":**
1. Persistent Homology (automated detector)
2. Potential Game framing (reframes problem as equilibrium selection)
3. MoE / role-aware (MaPE-style heterogeneity)

**Топ-3 по "publishable novelty":**
1. NEB-AGM hybrid (этот документ)
2. PH-based same-basin detector standalone
3. Potential Game perspective paper standalone

---

## Часть 9. Concrete experiments (validation roadmap)

### 9.1. Toy potential validation
- **Müller-Brown 2D:** classic NEB benchmark
- Implement NEB-AGM
- Compare: standard NEB vs CI-NEB vs string vs NEB-AGM
- Metric: convergence rate, robustness к bad initial path

### 9.2. Persistent Homology same-basin detector
- Standalone tool: take any converged NEB output, run PH diagnostic
- Run on **our actual mack/pent/marc data** (already harvested)
- Verify: PH bottleneck < threshold для same-basin cases (retrospective validation)
- Publishable: short methods note "Persistent Homology as a Diagnostic for NEB Same-Basin Artifacts"

### 9.3. Potential Game perspective paper
- No new algorithm, pure reframing
- Show that:
  - NEB is formally a potential game
  - Same-basin trap = Nash equilibrium selection failure
  - Suggest mechanism design fixes
- Target: J Chem Phys "Perspective" article

### 9.4. Full NEB-AGM на Fe-S V_Fe benchmark
- Implementation на нашем pipeline (ASE + QE)
- Benchmark на mack/pent/marc/greig/marc/pyr
- Honest comparison vs standard CI-NEB
- If wins → algorithmic paper J Chem Theory Comp

---

## Часть 10. Publication strategy

**Three-paper potential:**

### Paper A: Diagnostic (low-risk, fast)
- **Title:** "Persistent Homology as a Diagnostic for Same-Basin Artifacts in Nudged Elastic Band Calculations"
- **Format:** Methods note / short paper
- **Time:** 2-3 months
- **Risk:** Low (clean methodological contribution)
- **Target:** J Chem Theory Comp or J Chem Phys Communications

### Paper B: Perspective (medium-risk, fast)
- **Title:** "Game-Theoretic Foundations of Nudged Elastic Band Methods"
- **Format:** Perspective / Position paper
- **Time:** 4-6 months
- **Risk:** Medium (reviewers may ask for new algorithm)
- **Target:** Annu Rev Phys Chem или J Phys Chem Lett Perspective

### Paper C: Algorithm (high-risk, slow)
- **Title:** "Adaptive Game-Theoretic Nudged Elastic Band with Per-Image Memory (NEB-AGM)"
- **Format:** Full methodology paper + benchmarks
- **Time:** 12-18 months
- **Risk:** High (must beat existing methods convincingly)
- **Target:** J Chem Theory Comp or Comm Phys

**Sequence:** A → B → C. Каждый последующий builds on credibility previous. Total = 18-27 months for 3-paper sequence.

---

## Часть 11. Honest self-critique

### 11.1. Where the framing is solid

- Potential game framing: rigorous, verifiable, just needs writing
- ADMM reformulation: mathematically clean, parallels existing literature
- PH diagnostic: directly applicable, low risk

### 11.2. Where I might be over-selling

- **Mean-Field Game connection:** elegant but practical implementation для DFT is hard. Could be hype.
- **NEB-AGM as monolithic algorithm:** combines many components, debugging будет nightmare. Modular implementation safer.
- **"Equilibrium selection" framing:** Harsanyi-Selten theory developed for finite normal-form games, application к continuous strategy spaces is non-trivial.

### 11.3. Where prior art может убить

- **Spring Pair Method (Cao 2024, lit-scan §6.5):** уже 2-agent с явной communication. Если кто-то обобщит до N-agent, наша game-theoretic perspective становится derivative.
- **String method:** "chain of replicas" formalism уже есть. Reviewer-1: "это просто string method с тегами теории игр".
- **Replica Exchange NEB:** упоминалось 2010-х. Если активно развивается — конкурент.

### 11.4. Conservative recommendation

**Не пытаться сделать все 3 paper сразу.** Начать с Paper A (PH diagnostic) — это **standalone polishable result**, который validates approach. Если PH detector работает на нашем dataset → confidence boost для Papers B и C.

Paper B и C — **only after main Third Matter paper submitted**. Не distraction.

---

## Часть 12. Open questions

1. **Convergence theory:** какие условия на V гарантируют convergence NEB-AGM? Открытая теоретическая проблема.

2. **Shapley computation cost:** MC estimation требует O(N²) evaluations per round. Acceptable для N=9? Approximation needed?

3. **Role transition stability:** что предотвращает images от oscillating между ролями каждую iteration? Нужна regularization/hysteresis.

4. **PH sensitivity:** какой threshold для bottleneck distance? Должно зависеть от system. Universal threshold существует?

5. **Connection с TPT:** committor function через MFG-OT-TPT треугольник — formal proof? Could give rate constants directly.

6. **Mechanism design dictionary:** какие конкретные role_bonus terms доказуемо break degenerate Nash equilibria?

7. **Empirical question:** работает ли это лучше CI-NEB на FeS V_Fe? **Without empirical win, theory не публикуема.**

---

## Часть 13. Reading list (приоритезированный)

### Tier 1 (read first, 2-3 weeks)
- Boyd et al. 2011, "Distributed Optimization via ADMM" (Foundations and Trends в ML)
- Monderer & Shapley 1996, "Potential Games" (Games Econ Behav)
- Lasry & Lions 2007, "Mean field games" (Japanese J Math)
- Edelsbrunner & Harer "Computational Topology: An Introduction" (Ch 1-3 для PH basics)

### Tier 2 (foundational, 1-2 months)
- Toselli & Widlund "Domain Decomposition Methods" (selected chapters)
- Hazan "Introduction to Online Convex Optimization" (textbook)
- Harsanyi & Selten 1988 "A General Theory of Equilibrium Selection" (sections on risk dominance)
- Tassa, Mansard, Todorov 2014 "Control-Limited DDP" (IROS)

### Tier 3 (specialized)
- Wang et al. 2019 "ADMM for nonconvex problems" (Math Prog)
- Castro et al. 2009 "Polynomial calculation of Shapley value" (Comput Oper Res)
- Hiraoka et al. 2016 "Hierarchical structures in amorphous solids" (PNAS) — PH в materials
- Carmona & Delarue "Probabilistic Theory of Mean Field Games" (Vol I-II) — для MFG depth

---

## Часть 14. Cross-references

**Внутри проекта:**
- `ALTERNATIVES_AND_ROLE_AWARE_NEB.md` (companion document, landscape + lit-scan)
- the empirical same-basin diagnosis notes (MACK/PENT NEB protocol)
- the OPES pilot tracker (orthogonal альтернатива — OPES)
- the cross-mineral V_Fe barrier pattern notes (паттерн through Fe-S minerals)
- the hand-crafted same-basin endpoint detector (formalize через PH)

**Внешние якоря:**
- ByteDance Lance / MaPE 2026 (вдохновение)
- Boyd ADMM (math foundation)
- Monderer-Shapley potential games (game-theoretic foundation)
- Edelsbrunner-Harer PH (topological diagnostic)

---

## Часть 15. Статус

- **Тип:** research direction proposal + framework + concrete formulation
- **Уровень уверенности:**
  - Potential game framing: HIGH (just verify and write)
  - PH diagnostic: HIGH (testable on existing data)
  - NEB-AGM full algorithm: MEDIUM (needs prototype)
  - MFG-OT-TPT triangle: SPECULATIVE (deep math, hard prove)
- **Когда возвращаться:** после Third Matter main paper submitted
- **Кто должен ревьюить:** mathematician (game theory rigor), computer-scientist (algorithm complexity), physicist (NEB practical),  statmech-theorist (TPT/MFG connection)
- **Ближайшее действие (после current priorities):** PH diagnostic prototype на existing mack/pent/marc data — **standalone, low-risk, publishable independently**.

**Не реализовывать сейчас.** Parking lot для post-Third-Matter research program.
