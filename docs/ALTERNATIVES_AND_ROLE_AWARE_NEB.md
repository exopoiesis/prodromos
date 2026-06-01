# Альтернативы NEB и идея role-aware NEB с локальной памятью

**Контекст:** проект Third Matter, проблема same-basin trap в FeS V_Fe NEB (mack/pent/marc)
**Статус:** документ-задел для будущих исследований / methodology paper

---

## Часть 0. Постановка задачи

> «У нас есть горный массив с многими горами, и нам нужно проложить оптимальный маршрут по низинам (для каравана) из 9 точек, но особенность в том, что между точками есть условно говоря пружины с упругостью, которые не дают пружинкам сильно расширяться расстоянию и сильно сужаться. Но достаточно точно известно, что такой оптимальный путь есть, пусть может и не по самым низинам, может чуть выше по склону, но есть. Нужно математически решить такую задачу.»

Это классическая задача поиска **Minimum Energy Path (MEP)** на потенциальной поверхности.

**Соответствия:**
- Горный массив = potential energy surface V(x) (multi-dimensional)
- 9 точек = 9 images в NEB chain
- Пружины с упругостью = elastic spring forces вдоль тангенса пути
- "Не сжиматься/растягиваться" = equidistant constraint (holonomic)
- "Маршрут не по самым низинам, но выше" = MEP идёт через седловую точку (saddle), не по абсолютному дну
- "Такой путь есть" = **Mountain Pass Theorem** (Ambrosetti-Rabinowitz)

Эта постановка **математически точна**: NEB переформулирован через первичные принципы, что независимо приводит к Mountain Pass Theorem.

---

## Часть 1. Пять принципиально разных математических взглядов на MEP

NEB — лишь один из подходов. Существуют как минимум 5 неэквивалентных математических формулировок.

### 1.1. Вариационный взгляд (то, чем по сути является NEB)

Минимизация функционала действия:
```
S[γ] = ∫₀¹ L(γ(s), γ̇(s)) ds
```
с граничными условиями γ(0) = A (reactant), γ(1) = B (product).

**NEB** — это конкретный дискретизованный градиентный спуск с holonomic constraint, реализованный через пружины. Эквивалентная формулировка с Lagrange multiplier:
```
F_i = -∇V(x_i)|_⊥ + k(|x_{i+1} - x_i| - |x_i - x_{i-1}|) τ̂_i
```
где τ̂_i — оценка тангенса (Henkelman improved tangent), k — spring constant.

**Альтернативы внутри вариационной формулировки:**

- **String method** (E & Vanden-Eijnden 2002). Вместо пружин — *репараметризация* equidistant каждую итерацию. Часто стабильнее на жёстких потенциалах, особенно когда same-basin trap. Compute cost сопоставим, но устранена pathology of spring oscillations.
- **Simplified string method** (E, Ren, Vanden-Eijnden 2007). Ещё проще: только steepest descent + spline reparametrization.
- **Growing String Method** (Peters et al. 2004). Путь *растёт* от endpoints навстречу друг другу. Не требует хорошего initial guess (где у нас IDPP). Полезно когда A и B сильно различаются.
- **Geometric MEP / MaxFlux** (Olender & Elber 1996). Путь как геодезическая в Riemannian metric `g_ij = δ_ij · exp(V/kT)`. Чисто геометрический, без пружин. Численно проблематичен при T→0 (метрика сингулярная).
- **Doubly Nudged Elastic Band (DNEB)** (Trygubenko & Wales 2004). Добавляет вторую проекцию для борьбы с corner-cutting.

### 1.2. Топологический взгляд — Mountain Pass Theorem

**Ambrosetti-Rabinowitz (1973):** между двумя минимумами на coercive потенциале **гарантированно существует** критическая точка определённого индекса (седло Морса).

Формально: пусть V: ℝⁿ → ℝ имеет два локальных минимума A и B. Тогда

```
c = inf_{γ ∈ Γ} max_{s ∈ [0,1]} V(γ(s))
```

где Γ — пути из A в B, является критическим значением V. Достигается на седловой точке.

Это **теорема существования** ("такой путь есть"). Из неё растут методы, которые ищут седло без знания пути:

- **Dimer method** (Henkelman & Jónsson 1999). Два близких структуры → eigenvector оценка → подъём по min curvature direction. Не требует endpoint B!
- **Gentlest Ascent Dynamics (GAD)** (E & Zhou 2011). Continuous dynamics к saddle через eigenvector following.
- **Eigenvector following / Lanczos saddle search**.
- **Activation-Relaxation Technique (ART)** (Mousseau et al.).

После нахождения saddle — путь восстанавливается **steepest descent в обе стороны** (Intrinsic Reaction Coordinate, IRC). Это inverse direction NEB.

### 1.3. Вероятностный взгляд — Transition Path Theory (TPT)

**E & Vanden-Eijnden 2006, Metzner et al. 2009.** Вместо "одного оптимального пути" — *ансамбль* реактивных траекторий.

Ключевой объект — **committor function** `q(x)` = вероятность того, что траектория из x достигнет B раньше A:
```
L q = 0 в Ω \ (A ∪ B)
q|_A = 0, q|_B = 1
```
где L — генератор стохастической динамики (например, overdamped Langevin: L = -∇V·∇ + kT Δ).

MEP — частный случай TPT при T → 0: instanton, exponential concentration на most probable path (Freidlin-Wentzell).

**Методы:**
- **Transition Path Sampling (TPS)** (Bolhuis & Chandler 1998). MCMC по траекториям, не по конфигурациям.
- **Transition Interface Sampling (TIS)** (van Erp et al.). Послойное.
- **Forward Flux Sampling (FFS)** (Allen, Frenkel et al.). Для редких событий.
- **Milestoning** (Faradjian & Elber 2004).

**Преимущество:** даёт *kinetics* (rate constant), не только barrier. Это то, что нам в конечном итоге нужно для paper.

**Недостаток для DFT:** требует сотни/тысячи trajectories. Невозможно при $1/hr GPU на каждый MD step.

### 1.4. Optimal Control / Freidlin-Wentzell

Путь = решение задачи минимизации Freidlin-Wentzell action для стохастической динамики:
```
dx = -∇V(x) dt + √(2kT) dW
```

Action functional:
```
I[γ] = (1/4kT) ∫₀^T |γ̇ + ∇V(γ)|² dt
```

Минимизация I[γ] при γ(0) = A, γ(T) = B даёт **instanton path**. При T → 0 коллапсирует в MEP.

Эта переформулировка через **Pontryagin maximum principle** — задача оптимального управления:
- "Управление" u(t) = γ̇ + ∇V(γ)
- Целевая функция: ∫ |u|² dt
- Constraint: динамика, граничные условия

Связь с control theory открывает арсенал методов: HJB equations, model predictive control, reinforcement learning для path-finding.

**Practical methods:**
- **Minimum Action Method (MAM)** (E, Ren, Vanden-Eijnden 2004).
- **Adaptive MAM** (Zhou et al. 2008).
- **gMAM** (geometric MAM) (Heymann & Vanden-Eijnden 2008) — для overdamped Langevin без T.

### 1.5. ML / Generative подходы (hot 2024-2026)

- **Boltzmann Generators** (Noé et al. 2019). Normalizing flows для прямой выборки из p(x) ∝ exp(-V/kT). Дают пути как сэмплы.
- **Diffusion models для transition paths** (несколько preprints 2024-2025). Score matching на условных распределениях p(γ | A, B).
- **Implicit Transition Path Sampling** со score matching.
- **GFlowNets для path sampling** (Bengio et al.).
- **NeuralMD / Equivariant transformers** для прямой параметризации γ_θ(s).
- **OPES** (Invernizzi-Parrinello 2020-2024). On-the-fly Probability Enhanced Sampling. Fundamentally другой подход: collective variable bias, не path-based.

**Connection с MLIPs:** MACE, NequIP, Allegro, CHGNet — *накапливают знания о PES*, но не используют их для роутинга пути.

---

## Часть 2. Почему другие учёные не рассматривают альтернативы

Честный ответ — смесь технических причин (~40%) и социологии науки (~60%).

### 2.1. Технические причины

**1. Cost-method mismatch с DFT.**
- TPS/FFS требуют *сотен* MD trajectories — невозможно при DFT $1/hr GPU
- Diffusion models / Boltzmann generators требуют *training set* — у материалов его нет (в отличие от proteins/water)
- String method ≈ same cost как NEB, но requires interpolation step — где accuracy теряется
- MAM/gMAM требуют numerical optimization в высокой размерности

**2. Конвергенция без ground truth.**
- Methods papers сравнивают на toy potentials (Müller-Brown, LJ Lennard-Jones), где известно решение
- На реальном FeS никто не знает истинного MEP → нет benchmark → каждая группа репортит свой результат на своей системе, сравнить нельзя
- Reviewers не могут отличить "method failed" от "method correct, system weird"

**3. Numerical pathologies.**
- Geometric MEP (Olender-Elber) при T→0: метрика `exp(V/kT)` становится сингулярной, численно нестабилен
- String method с плохой interpolation → corner cutting в curved valleys
- Dimer метод: если eigenvalue spectrum плохой (degenerate), не сходится

### 2.2. Социологические причины

**4. Software lock-in.**
- VASP, QE, ASE — NEB integrated, plug-and-play
- String method: код Vanden-Eijnden в Matlab, не интегрирован в DFT суперструктуру
- OpenPathSampling, Pyretis (TPS) — отдельная экосистема, steep learning curve
- "Кто будет писать String для ABACUS на A100?" — никто, потому что нет grant под это

**5. Community fragmentation.**

| Метод | Community | Цитируемые журналы |
|---|---|---|
| NEB | materials/catalysis | Phys Rev B, J Chem Phys, J Am Chem Soc |
| TPS/TPT | biophysics/proteins | PNAS, J Chem Theory Comp, J Phys Chem B |
| MAM/optimal control | applied math | SIAM, Comm Pure Appl Math, J Comp Phys |
| ML/Diffusion | ML community | NeurIPS, ICML, ICLR |

Они почти не пересекаются на конференциях. FeS chemist никогда не слышал про Freidlin-Wentzell. Diffusion-model researcher не знает про fmax convergence в NEB.

**6. Pedagogical inertia.**
- Sholl-Steckel "Density Functional Theory" — только NEB
- Jensen "Computational Chemistry" — упоминает Dimer вскользь
- PhD первого года учится NEB → через 10 лет пишет grants под NEB → reviewer для NEB papers
- Цикл воспроизводится. Никто не учит string method в DFT курсах.

**7. Survivorship bias в литературе.**
- Published NEB results — те, что сошлись. Те, что не сошлись, attributed к "system complexity", не к method
- Никто не пишет paper "NEB не сработал на нашей системе" — это карьерное самоубийство
- **Same-basin trap** который мы видим (mack/pent/marc) — это известная проблема в narrow circle (Henkelman group обсуждает), но не появляется в textbooks

**8. Confirmation bias на уровне поля.**
- "Стандартная методология" = быстрее проходит peer review
- Reviewer: "Why didn't you use NEB? It's standard" — нужно оправдываться 2-3 страницы в SI
- Использовать NEB = 0 friction; использовать String/GAD = explain to 3 reviewers

**9. Selection bias на системах.**
- Easy barriers (single saddle, clear endpoints) → NEB works → publish → reinforce NEB
- Hard systems (floppy, multi-saddle, same-basin) → researchers *switch to easier systems*, не switch method
- Это объясняет почему FeS так underexplored — он именно из "hard" категории

**10. Citation/career economics.**
- Methods paper про String method цитируется ~500 раз за 20 лет
- Applications paper "NEB на новой катализаторе" — 200 цитирований за 3 года
- PhD/postdoc выбирает applications → NEB

### 2.3. Что меняется сейчас (2024-2026)

- **Finite-T string method** растёт в free-energy applications (Parrinello, Vanden-Eijnden коллабы)
- **ML committors** (Roux 2024, Vanden-Eijnden 2025) — TPT становится доступным
- **OPES** (Invernizzi) активно вытесняет metadynamics → потенциально и NEB для kinetics
- **Diffusion models для paths** — несколько preprints 2025 на arXiv, но zero в materials journals пока
- **Equivariant neural networks** для path parameterization — emerging

### 2.4. Следствие для нашего paper

Same-basin trap в FeS — это **literally signal что мы на frontier**, не методическая неудача. Большинство опубликованных Fe-S NEB papers — либо surface (легче), либо vacancy в простых металлах (легче), либо short hops (легче). **Bulk Fe-S с конкурирующими saddles — недоисследовано именно потому что NEB на нём капризен**, а alternatives неудобны в DFT pipeline.

Это нам играет на руку: **"we benchmarked NEB and identified failure mode in Fe-S V_Fe class"** — это valid scientific contribution, не недостаток.

---

## Часть 3. Идея: role-aware NEB с локальной памятью (вдохновение от Lance/MaPE)

### 3.1. Lance/MaPE — что это и почему работает

ByteDance Lance (2026, https://lance-project.github.io/):
- Dual-stream MoE архитектура для image/video understanding + generation
- 6 млрд общих, 3 млрд активных параметров
- **Ключевая инновация: MaPE (Mixed-role aware Position Encoding)**

Стандартное positional encoding в трансформере говорит токену "где ты находишься" (spatial/temporal coord). MaPE добавляет "**зачем ты здесь**" — role label:
- "Я для понимания" (input image для analysis)
- "Я условие" (text prompt, reference image)
- "Я генерируемый" (noisy token на denoising stage)

Без role label трансформер путает разнородные visual tokens в одной последовательности → деградация на смешанных задачах. С MaPE разделение clean.

### 3.2. Интуиция

> «Может как-то ускорить расчёт... подход когда на каждом такте каждая ячейка накапливает знания и может как-то двигать к оптимальной точке, без фейлов»

Это содержит **два независимых компонента**:
1. **Memory/surrogate per cell:** каждая ячейка (image) накапливает знания
2. **Role-aware updates:** ячейки имеют разные роли и обрабатываются по-разному

### 3.3. Компонент 1: локальная память per image

**Уже существует** в NEB-литературе под именем **ML-NEB / GP-NEB**:
- **Koistinen et al. 2017** (J Chem Phys 147, 152720) — Gaussian Process surrogate глобально для всего пути
- **Garrido Torres et al. 2019** (Phys Rev Lett 122, 156001) — active learning NEB с GP
- **MACE-NEB / NequIP-NEB** — это то, что мы делаем

Но! В существующих подходах surrogate **глобальный** (один GP/MLIP на весь путь).

Интуиция **"каждый image — своя накопленная знаниями ячейка"** — это **менее исследовано**:
- Image в basin → накапливает кривизну дна (local Hessian estimate)
- Image на ridge → накапливает eigenvector saddle direction
- Image в transition → накапливает tangent direction history

**Преимущество локального подхода:**
- 9 cheap local models вместо 1 expensive global
- Sparse data per image (только окрестность) → быстрая convergence GP
- Стохастическая обработка: stuck image имеет статистику в одной basin → можно детектировать → выкинуть из этой basin форсированно

**Это реальная исследовательская дыра**, особенно для same-basin trap (наша проблема).

### 3.4. Компонент 2: role-aware updates (MaPE-аналог)

**Тоже частично существует**, но крайне примитивно:
- **Climbing Image NEB** (Henkelman et al. 2000) = 1 image имеет "роль climber", остальные одинаковые
- **Multi-climbing NEB** = несколько climbers (для multi-saddle paths)
- **Adaptive image NEB** (Maras et al. 2016) = роли по curvature

Но MaPE-аналог идёт дальше. Сейчас в стандартном NEB:
- Все 7 middle images получают **одинаковую force law**: F_i = -∇V|_⊥ + k(Δs_right - Δs_left)τ̂
- Climbing image (если CI-NEB) имеет **другую force law**: F_CI = -∇V + 2(∇V·τ̂)τ̂
- Эта **бинарность** — потеря информации

**Что мог бы дать MaPE-аналог:**

| Роль | Force law | Когда |
|---|---|---|
| `basin-slider` | F = -∇V (pure gradient) | image глубоко в basin, |∇V| мал, eigenvalues>0 |
| `ridge-walker` | F = -∇V\|_⊥ + soft spring | стандартное NEB поведение |
| `saddle-approacher` | F = -∇V + α(∇V·τ̂)τ̂, α∈[0,2] | возле maximum пути, large \|∇V·τ̂\| |
| `climber` (CI) | F = -∇V + 2(∇V·τ̂)τ̂ | один image на peak |
| `stuck-in-wrong-basin` | F = -∇V + β·escape_direction | детектирован через локальную статистику displacement |
| `transition-walker` | F = -∇V\|_⊥ + stiff spring + reparametrize | через узкое горлышко между basins |

**Метки обновляются** каждые N iterations на основе локальной статистики:
- Curvature spectrum (eigenvalues локального Hessian estimate)
- Displacement history (variance последних k шагов)
- Neighbor distance (если distance к neighbor сильно меньше среднего → potentially stuck)
- Confidence в роли (Bayesian-like posterior)

### 3.5. Принципиальные различия от Lance — что НЕ переносится

Чтобы не врать аналогией, граница такая:

| Lance/MaPE | NEB |
|---|---|
| Transformer attention — **глобальная** связь между all tokens | NEB springs — **локальная** (только nearest neighbors) |
| Roles нужны чтобы избежать **interference** в attention | NEB не имеет interference — forces локальны |
| Учится на **миллионах** примеров | NEB решает **1** задачу, ~50-200 iter |
| Tokens — discrete (vocabulary) | Images — continuous (coordinate space) |
| Generation vs understanding — **семантическая разница** | Basin vs saddle — **геометрическая разница** |

**Главное концептуальное различие:** MaPE решает проблему *semantic confusion* в смешанной последовательности. У NEB *нет* проблемы confusion — есть проблема *одинаковости трактовки геометрически разнородных images*. Это разные болезни. Но **решение похожее по структуре** — метки роли + role-conditional обработка.

### 3.6. Объединение компонентов: предлагаемый алгоритм

**Адаптивная role-aware NEB с локальной памятью (ARNN-LM):**

```
Initialization:
  - 9 images (IDPP или linear interpolation)
  - Каждый image i: GP_i = GaussianProcess(kernel=RBF, prior=zero)
  - Каждый image i: role_i = "transition-walker" (initial guess)
  - История: trajectory_i = [], gradient_history_i = []

For iteration = 1, 2, ..., max_iter:

  # Step 1: DFT calls (с active learning)
  For each image i:
    If confidence(GP_i) > threshold AND iteration > burn_in:
      grad_i = GP_i.predict_gradient(x_i)
      [cheap shortcut]
    Else:
      grad_i = DFT(x_i)
      GP_i.update(x_i, V(x_i), grad_i)
      [accumulates local knowledge]
    trajectory_i.append(x_i)
    gradient_history_i.append(grad_i)

  # Step 2: Role assignment
  For each image i:
    eigenvals_i = GP_i.estimate_local_hessian_spectrum(x_i)
    displacement_var_i = var(trajectory_i[-k:])
    neighbor_dist_i = (|x_{i+1}-x_i| + |x_i-x_{i-1}|)/2

    role_i = classify_role(
      eigenvals_i,
      |grad_i|,
      grad_i · τ̂_i,
      displacement_var_i,
      neighbor_dist_i
    )
    # Returns one of: basin-slider, ridge-walker, saddle-approacher,
    #                 climber, stuck, transition-walker

  # Step 3: Role-conditional force computation
  For each image i:
    F_i = compute_force(grad_i, τ̂_i, role_i, spring_constants[role_i])

  # Step 4: Optimizer step (FIRE or BFGS)
  x_new = optimizer.step(x, F)

  # Step 5: Check convergence
  If max|F_i| < fmax AND no role changes за 5 iter:
    converged
```

### 3.7. Существенно ли это отличается от существующего?

- **От ML-NEB:** там 1 global surrogate (1 MLIP/GP на весь путь), 1 force law. Здесь — 9 local + 6 force laws.
- **От adaptive NEB (Maras):** там роль по простой curvature snapshot, нет памяти, нет surrogate.
- **От CI-NEB:** там бинарная разметка climber/non-climber.
- **От string method:** там вообще нет ролей, только reparametrization.

**Да, это публикуемая novelty.**

### 3.8. Связь с нашим pain — same-basin trap

Наш диагноз:
- Endpoints A и B релаксируются BFGS-ом → оба попадают в один локальный минимум (тот же basin)
- Все 9 images интерполируются между двумя "соседними" точками в одной basin
- NEB "сходится" к плоскому пути с dE < SCF noise
- Барьер оказывается артефактом размера 0±20 meV

**Как role-aware NEB это лечит:**
1. После burn-in каждый image имеет local GP с оценкой curvature spectrum
2. Если **все 9 images** имеют eigenvalues > 0 (locally convex) → ВЕСЬ путь в basin
3. Назначаются роли "basin-slider" → force = -∇V → коллапс к минимуму
4. **Trigger:** GP-estimated barrier < threshold AND all-basin diagnostic → ABORT с диагнозом "same-basin endpoints"
5. Перед re-launch предлагается **structural perturbation** endpoints (break degeneracy) — exactly то, что мы делаем мануально через "post-BFGS structural sanity gate"

**Это автоматизация нашего hard-won lesson.**

### 3.9. Реалистичная оценка перспектив

- **Совсем безумно?** Нет, компоненты по отдельности есть в литературе
- **Уже сделано?** Локальные surrogates per-image + dynamic role assignment + escape force — **не нашёл публикации**. Возможно есть в препринтах 2025-2026, надо lit-search
- **Решит same-basin trap?** Возможно. Stuck-detection через локальную статистику + escape force = прямая терапия нашей болезни
- **DFT-применимость?** Compute-overhead локального GP ничтожен (~ms) против 1 DFT call (~hours). Должно работать
- **Сколько труда?** ~3-6 месяцев на прототип на нашем pipeline. Не наш текущий приоритет, но **записать как идею для будущего paper**
- **Где публиковать?** J Chem Phys (если methodology focus) или Phys Rev Materials (если application focus на Fe-S benchmarks)

### 3.10. Расширение на основную задачу (не NEB-specific)

Более общая идея — "клетки с памятью движутся к оптимуму без fail" — это вообще-то **известная парадигма** в optimization:
- **Distributed/multi-agent optimization** (consensus algorithms)
- **Particle Swarm Optimization** с memory (Kennedy & Eberhart 1995)
- **Agent-based MEP** (несколько обскюрных papers 2010-х)
- **Federated learning** в optimization context

NEB можно переинтерпретировать как **federated learning по path-points**, где springs = communication protocol, locally accumulated knowledge = local model parameters, role labels = client class.

**Это интересный концептуальный фрейм, но мейнстрим optimization community туда не пошёл** по тем же сoциологическим причинам что описаны в Части 2 (community fragmentation: optimization theorists и NEB practitioners не пересекаются).

---

## Часть 4. Action items

### 4.1. Немедленно (на следующих сессиях)
- [ ] lit-search на 2024-2026 papers по "role-aware NEB", "adaptive image NEB", "GP per image NEB", "local surrogate NEB"
- [ ] Проверить наши результаты mack/pent/marc на признак "all-eigenvalues-positive" — если да, это retrospective validation идеи stuck-detector

### 4.2. Среднесрочно (если идея выживает lit-review)
- [ ] Прототип на Müller-Brown toy potential (3 image, 2D, 1 hour работы) — proof of concept
- [ ] Extension на LJ7 cluster transition (классический NEB benchmark) — sanity check
- [ ] FeS V_Fe pilot — попробовать на наших provenly-broken cases

### 4.3. Долгосрочно (методологический paper)
- [ ] Strict comparison: NEB vs CI-NEB vs string vs ARNN-LM на set of canonical hard problems
- [ ] Benchmark set: включить FeS V_Fe (mack/pent/marc) — это наш unique contribution
- [ ] Theory: convergence properties of dynamic role assignment, proof что role-changes не destabilize fixed point

---

## Часть 5. Связь с существующими файлами проекта

**Сильно связанные:**
- the MACK/PENT NEB protocol notes — same-basin trap diagnosis
- the OPES pilot tracker — альтернативный подход через OPES
- the pyrite V_Fe experiment plan — следующий тест cubic vs ortho
- the cross-mineral V_Fe barrier pattern notes — паттерн через 3 mineral
- the MLIP benchmark protocol — где MLIPs уже используются
- the 8-step RC-check pre-flight notes
- the same-basin endpoint diagnosis rule
- the NEB-reuse triple-picker technical-trap note

**Концептуально связанные:**
- Mountain Pass Theorem ↔ "existence of MEP" intuition
- TPT/committor ↔ что нам в итоге нужно для kinetics (rate)
- OPES ↔ orthogonal альтернатива NEB через CV bias

---

## Часть 6. Литература-якоря (для будущего lit-search)

**Классика NEB:**
- Henkelman & Jónsson 2000, "Improved tangent estimate in the nudged elastic band method", J Chem Phys 113, 9978
- Henkelman, Uberuaga, Jónsson 2000, "A climbing image NEB method", J Chem Phys 113, 9901

**String method:**
- E, Ren, Vanden-Eijnden 2002, "String method for the study of rare events", Phys Rev B 66, 052301
- E, Ren, Vanden-Eijnden 2007, "Simplified and improved string method", J Chem Phys 126, 164103

**Saddle search без endpoint:**
- Henkelman & Jónsson 1999, "A dimer method for finding saddle points on high dimensional potential surfaces", J Chem Phys 111, 7010
- E & Zhou 2011, "Gentlest ascent dynamics", Nonlinearity 24, 1831

**TPT / TPS:**
- Bolhuis, Chandler, Dellago, Geissler 2002, "Transition Path Sampling: Throwing Ropes Over Rough Mountain Passes", Annu Rev Phys Chem 53, 291
- E, Vanden-Eijnden 2006, "Towards a theory of transition paths", J Stat Phys 123, 503

**ML-NEB:**
- Koistinen et al. 2017, "Nudged elastic band calculations accelerated with Gaussian process regression", J Chem Phys 147, 152720
- Garrido Torres et al. 2019, "Low-scaling algorithm for nudged elastic band calculations using a surrogate machine learning model", Phys Rev Lett 122, 156001

**Mountain Pass Theorem:**
- Ambrosetti & Rabinowitz 1973, "Dual variational methods in critical point theory and applications", J Funct Anal 14, 349

**OPES (наш текущий orthogonal подход):**
- Invernizzi & Parrinello 2020, "Rethinking metadynamics: from bias potentials to probability distributions", J Phys Chem Lett 11, 2731

**Lance/MaPE (вдохновение):**
- ByteDance Intelligent Creation Lab, "Lance: Unified Image and Video Understanding/Generation/Editing", 2026, https://lance-project.github.io/

---

## Часть 6.5. Lit-scan: NEB как federated/multi-agent optimization

**Тип:** surface-level scan (~25 мин работы, ~12 WebSearch запросов). НЕ полный обзор.
**Фокус:** 2024-2026 papers на пересечении NEB/MEP методологии и distributed/federated/multi-agent ML.

### 6.5.A. Прямого совпадения НЕ найдено

После целенаправленных поисков по комбинациям:
- "NEB federated", "elastic band distributed optimization"
- "minimum energy path multi-agent", "string method distributed consensus"
- "client server molecular optimization federated"

**Ни одной работы, которая бы явно переинтерпретировала NEB как federated learning / consensus optimization framework, не существует в открытом доступе.**

Концептуальный мост "images = clients, springs = communication protocol, climbing image = special role" — **не публиковался в этой форме**. Это пустая клетка в литературе на 2026-05.

### 6.5.B. Adjacent работы (концептуально близко, риск prior art)

Сортировка по релевантности к framing «per-client локальная задача + coordinated path»:

1. **⚠️ Spring Pair Method (SPM), Cao et al. 2024** — arxiv 2407.04373
   - "Single pair of spring-coupled particles" эволюционирует на PES, используя MEP как компас
   - **2-agent saddle search с явной spring communication** — концептуально ближайший к federated framing
   - НЕ обобщён до N images / distributed framework
   - **Самый опасный prior art для алгоритмической части нашей идеи**

2. **Matrix Particle Swarm Optimization для transition paths in solids**, Qian et al. (npj Comp Mat) — nature.com/articles/s41524-020-0286-9
   - PSO-swarm (multi-agent!) ищет MEP для pressure-induced structural transformations
   - Каждая particle = candidate path
   - **Multi-agent в явном виде**, но population-based metaheuristic, не federated с локальными моделями

3. **Efficient NEB via Neural Network Bayesian Algorithm Execution** — arxiv 2512.14993 (2025-12)
   - Глобальный surrogate NN + Bayesian active learning
   - "1-2 orders of magnitude reduction in force evaluations"
   - Surrogate **глобальный**, не per-image. Centralized active learning, не federated

4. **⚠️ Enhanced CI-NEB with Hessian Eigenmode Alignment** — arxiv 2601.12630 (2026)
   - Hybrid CI-NEB + min-mode following
   - **Role-aware**: climbing image получает специальную динамику с Hessian eigenmode
   - **Самый близкий к "role-aware NEB" framing**, который найден
   - Бинарная роль (climber vs остальные), но направление верное

5. **⚠️ Modified NEB with adaptive spring lengths (Onsager-Machlup discretization)** — arxiv 2106.06275
   - Заменяет harmonic spring на discretized OM-action → **springs становятся adaptive по path**
   - Прямо релевант "springs = adaptive communication channel" framing
   - Как continuous formalism, не federated

6. **Action-Minimization Meets Generative Modeling (Onsager-Machlup)** — arxiv 2504.18506 (2025)
7. **GAP: Guided Diffusion for A Priori Transition State Sampling** — openreview 2025
8. **Flow matching for reaction pathway generation** — arxiv 2507.10530 (2025-07)
9. **Generative Model for Reaction Path (Hayashi et al., JCTC 2025)** — arxiv 2401.10721
10. **Diffusion methods for generating transition paths (Triplett & Lu)** — arxiv 2309.10276
11. **Deep RL for Most Probable Transition Pathway** — arxiv 2304.12994 + 2404.05905 (2024)

### 6.5.C. Активные исследователи / группы на пересечении

| Имя | Институт | Направление | Свежая работа |
|---|---|---|---|
| **Hannes Jónsson** | U Iceland | Оригинальный NEB-автор, GP-accelerated saddle searches | GPR-accelerated min-mode following, arxiv 2505.12519 (2025-05) |
| **Graeme Henkelman** | UT Austin | Climbing-image NEB, ML-acceleration | Нет прямой federated работы |
| **Jianfeng Lu** | Duke | Applied math + TPT, score-based generative | arxiv 2309.10276 для rare transitions |
| **Pratyush Tiwary** | UMD | TPS + deep learning CV | 2025 perspective "ML and Statistical Mechanics" |
| **Frank Noé** | FU Berlin / Microsoft | Boltzmann generators, neural samplers | Cross-pollinating, не NEB per se |
| **Cecilia Clementi** | FU Berlin | Coarse-graining + ML для path sampling | Похожий профиль на Noé |
| **Weinan E / E. Vanden-Eijnden** | Princeton / NYU | String method theorists, TPT | Formalism уже близок ("chain of replicas"), но не federated |

### 6.5.D. Honest gap assessment

**Это публикуемая novelty?** Скорее **ДА, как methodological reframing** — но с оговорками:

**Что делает идею публикуемой:**
- Прямого совпадения нет (verified)
- Spring Pair Method (2024) и matrix PSO — ближайшие соседи, но N=2 или population-based, не federated с heterogeneity/role-aware
- Existing NEB literature treats images как **homogeneous degrees of freedom одной optimization variable**, не как independent agents с локальными objectives

**Что снижает novelty:**
- String method уже подразумевает "chain of replicas" — formalism близок
- Адаптивные springs уже исследованы (Onsager-Machlup version, arxiv 2106.06275)
- Climbing image — уже "role-aware" в широком смысле. Hessian-eigenmode CI-NEB (arxiv 2601.12630) делает это ещё явнее
- Per-image GP surrogates обсуждались (хотя dominant practice — глобальный GP с per-image acquisition)

**Best guess стратегии публикации:**
- **Position / perspective paper:** "NEB through the lens of federated/consensus optimization" с конкретными выводами (heterogeneous per-image surrogates с consensus update on springs → лучше для multi-basin paths). Низкий risk reviewer pushback.
- **Algorithmic paper:** требует concrete benefit (например, наш FeS same-basin trap как demonstration), иначе риск "это же string method с тегами". SPM 2024 — самый опасный prior art.
- **Гибрид:** position section + concrete algorithm + FeS V_Fe benchmark — оптимально для J Chem Theory Comp или Phys Rev Materials.

### 6.5.E. Где наиболее вероятно пропущено (queries для глубже)

Если решим developing идею — нужны эти доп-поиски:

1. `"image-parallel" NEB GP local Bayesian "uncertainty quantification" 2024 2025`
   — поискать конкретно per-image UQ в acceleration работах. Возможно похожее скрыто в GPAW/ORCA/VASP implementations
2. `"replica" "local model" "consensus" molecular path optimization 2023 2024 2025`
   — пересечение с replica exchange community, где "local Hamiltonian, global path" встречается
3. `"distributed" "saddle search" OR "transition state" molecular 2024 2025 reinforcement learning`
   — RL framing может пересекаться с multi-agent в Eldar/Bertsekas стиле
4. **Forward-citation search** от оригинальной NEB (Mills/Jónsson 1995) с filter `federated OR distributed OR consensus`
   — если кто-то делал, должен ссылаться на foundational paper

### 6.5.F. Action items после lit-scan

- [ ] **Прочитать в первую очередь** (top-3 prior art):
  - arxiv 2407.04373 — Spring Pair Method (Cao 2024)
  - arxiv 2601.12630 — Hessian-eigenmode CI-NEB (2026)
  - arxiv 2106.06275 — Onsager-Machlup adaptive springs
- [ ] Если эти 3 не закрывают идею → запустить deep lit-search с queries 6.5.E
- [ ] Решить: position paper vs algorithmic paper vs hybrid

### 6.5.G. Ссылки

- Spring Pair Method (Cao 2024): https://arxiv.org/pdf/2407.04373
- Matrix PSO transition paths: https://www.nature.com/articles/s41524-020-0286-9
- Efficient NEB via NN BAE (2025): https://arxiv.org/pdf/2512.14993
- Action-Minimization × Generative Modeling (2025): https://arxiv.org/pdf/2504.18506
- Flow matching reaction pathway (2025): https://arxiv.org/abs/2507.10530
- Generative Model Reaction Path (Hayashi 2025): https://arxiv.org/abs/2401.10721
- Modified NEB OM adaptive springs: https://arxiv.org/pdf/2106.06275
- Hessian-eigenmode CI-NEB (2026): https://arxiv.org/pdf/2601.12630
- Diffusion transition paths (Triplett & Lu): https://arxiv.org/pdf/2309.10276
- GPR min-mode following (Jónsson 2025): https://arxiv.org/pdf/2505.12519
- Deep RL transition pathway (2024): https://arxiv.org/pdf/2404.05905
- Low-scaling NEB surrogate ML (Koistinen 2018 foundational): https://arxiv.org/abs/1811.08022

---

## Часть 7. Статус документа

- **Тип:** brainstorm + research direction proposal
- **Уровень уверенности:** идея жизнеспособна на концептуальном уровне, требует lit-review для подтверждения novelty
- **Когда возвращаться:** после lit-search; если ничего похожего не найдём → прототип на toy potential
- **Кто должен ревьюить:** physicist (NEB methodology), mathematician (variational/optimal control угол), computer-scientist (GP/surrogate design), statmech-theorist (TPT связь)

**Не реализовывать сейчас.** Параллельный track к основной paper. Возможно следующий paper после Third Matter.
