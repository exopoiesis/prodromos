Важное уточнение: SPM, OM/adaptive springs, dyNEB, OCINEB, GP-NEB — это не "тулы" уровня QE/ABACUS/MACE/CHGNet/jDFTx. Это в основном
  алгоритмические варианты NEB / saddle search / ускорители path optimization. Они сидят над energy/force calculator.

  То есть схема такая:

  QE / ABACUS / jDFTx / MACE / CHGNet
          дают E(R), F(R)
                ↓
  NEB / string / dyNEB / GP-NEB / OCINEB / наш NEB-AGM
          двигают images/path
                ↓
  pre-flight verdict: GO / NO-GO / INVESTIGATE

  Для нашей цели они нужны не как "перейти на другой софт", а как карта известных математических решений, чтобы понимать: что уже существует,
  что можно украсть как идею, а где наша задача другая.

  Главное
  Цель сейчас правильнее, чем "превзойти всех":
  не запускать дорогой DFT NEB, пока математически/топологически не проверено, что input имеет смысл.

  То есть мы строим не "самый быстрый NEB", а pre-flight certificate / fail-fast framework.

  ———

  ## 1. SPM — Spring Pair Method

  Источник: Cui & Jiang, 2024, "A spring pair method of finding saddle points using the minimum energy path as a compass"
  https://arxiv.org/abs/2407.04373

  Что это.
  Метод поиска saddle point через две spring-coupled particles, а не через целую NEB-цепочку из 7-15 images.

  Обычный NEB:

  A -- img1 -- img2 -- ... -- imgN -- B

  SPM:

  particle 1 -- spring -- particle 2

  Эта пара движется по PES так, чтобы:

  - приблизиться к MEP,
  - сориентироваться вдоль tangent MEP,
  - использовать это направление для подъёма к saddle.

  Зачем придумали.
  Обычные saddle-search методы часто требуют Hessian/eigenvector или могут пойти к нерелевантному saddle. SPM пытается получить направление
  saddle через локальную геометрию MEP, без Hessian.

  Что нужно на входе.

  - PES oracle: энергия и градиент/силы.
  - Это может быть DFT, MLIP, analytic toy PES.
  - Не обязательно требует полного endpoint-pair NEB.

  Для нас полезно как pre-flight?
  Да, но ограниченно.

  Полезно:

  - дешёвый MLIP-probe: "есть ли вообще nearby saddle/перевал в этом кармане?"
  - альтернатива dimer/MMF для локального saddle search после endpoint screen.
  - хороший hard-toy baseline для нашего role-aware logic.

  Не решает:

  - same-basin endpoint collapse;
  - магнитные листы;
  - вопрос "A и B действительно разные basins?";
  - multi-endpoint network.

  Вывод для проекта:
  SPM не заменяет наш framework. Но его стоит держать как локальный saddle-probe после L1/L2, особенно на MACE/CHGNet, если endpoint-pair
  подозрительный.

  ———

  ## 2. OM/adaptive springs — Onsager-Machlup adaptive spring NEB

  Источник: Mandelli & Parrinello, 2021, "A modified nudged elastic band algorithm with adaptive spring lengths"
  https://arxiv.org/abs/2106.06275

  Что это.
  В стандартном NEB есть harmonic spring term: images примерно равномерно распределяются по пути.

  OM/adaptive springs меняет spring term: вместо обычной пружинной энергии используется дискретизированный Onsager-Machlup action. В
  результате images автоматически сгущаются около saddle / важной области пути.

  Обычная проблема NEB:

  images равномерны по path length,
  но saddle может попасть между images

  OM/adaptive springs:

  больше resolution там, где path важнее

  Что нужно на входе.

  - endpoint A/B;
  - energy/forces;
  - обычный NEB-like workflow.

  Для нас полезно как pre-flight?
  Да, как проверка path resolution, но не как endpoint-validator.

  Полезно:

  - если cheap MLIP NEB даёт разный barrier при разном image distribution, это warning;
  - если saddle появляется только при adaptive springs, обычный NEB был плохо параметризован;
  - идея близка к нашему adaptive-reparam в NEB-AGM.

  Не решает:

  - если A и B в одном basin, OM всё равно красиво оптимизирует бессмысленный путь;
  - если MLIP уходит в 20 eV яму, OM не спасает;
  - если spin sheet split, OM не видит магнетизм.

  Вывод:
  Для наших целей OM/adaptive springs — это не главный метод, а sanity check: "зависит ли вывод от распределения images?". Если зависит — DFT
  запускать нельзя.

  ———

  ## 3. dyNEB — dynamic NEB

  Источник: Lindgren, Kastlunger, Peterson, 2019, "Scaled and Dynamic Optimizations of Nudged Elastic Bands"
  https://arxiv.org/abs/1906.10257
  ASE docs: https://ase-lib.org/ase/mep.html

  Что это.
  dyNEB экономит force calls за счёт того, что не все images одинаково важны на каждой итерации.

  Если image уже почти сошёлся, его можно не оптимизировать так активно. Если image около saddle, ему можно уделить больше внимания.

  ASE имеет DyNEB, то есть это ближе к настоящему usable tool, чем SPM.

  Что нужно на входе.

  - обычные NEB images;
  - ASE-compatible calculator;
  - можно потенциально использовать с MACE/CHGNet/QE через ASE-слой.

  Для нас полезно как pre-flight?
  Скорее как cost-reduction после того, как input уже прошёл gate.

  Полезно:

  - MLIP preflight NEB быстрее;
  - DFT NEB дешевле, если уже уверены в endpoints/path;
  - хороший baseline для "не все images одинаковы", близко к role-aware идее.

  Не решает:

  - не доказывает, что endpoints разные;
  - не ловит топологический same-basin artifact сам по себе;
  - не лечит неправильный reaction coordinate.

  Вывод:
  dyNEB — это операционная оптимизация, не epistemic gate. Его можно использовать, но только после наших L0-L3 проверок.

  ———

  ## 4. OCINEB / Hessian Eigenmode Alignment

  Источник: Goswami et al., 2026, "Enhanced Climbing Image Nudged Elastic Band method with Hessian Eigenmode Alignment"
  https://arxiv.org/abs/2601.12630

  Что это.
  Гибрид:

  CI-NEB находит примерно saddle region
          ↓
  minimum-mode following / Hessian eigenmode alignment
          ↓
  уточняется saddle point

  То есть это не столько "pre-flight", сколько post-NEB saddle refinement.

  Идея: climbing image хорошо указывает релевантный transition region, но может медленно сходиться или застревать. Тогда включают MMF/dimer-
  like refinement, используя направление минимальной кривизны.

  Что нужно на входе.

  - уже есть NEB/CI-NEB path;
  - нужно уметь оценивать min-mode / Hessian-like direction;
  - с DFT это дорого, с MLIP дешевле.

  Для нас полезно как pre-flight?
  Косвенно.

  Полезно:

  - после cheap MLIP path можно проверить: "climber действительно ведёт к нормальному saddle?"
  - если OCINEB/MMF уходит в другой saddle, значит исходный path неустойчив;
  - можно использовать как validator для top-1/top-2 transition candidates.

  Не решает:

  - endpoint same-basin collapse до NEB;
  - магнитные discontinuities;
  - multi-site endpoint ambiguity.

  Вывод:
  OCINEB — сильный prior art против claim "мы первые сделали role-aware climber". Но для нашей цели он просто говорит: climber refinement уже
  известен, значит нам надо фокусироваться не на climber, а на до-DFT диагностике input topology.

  ———

  ## 5. GP-NEB — Gaussian Process accelerated NEB

  Источники:
  Koistinen et al., 2017, "Nudged elastic band calculations accelerated with Gaussian process regression"
  https://arxiv.org/abs/1706.04606
  Garrido Torres et al., 2019, "Low-Scaling Algorithm for NEB Using a Surrogate ML Model"
  https://pubmed.ncbi.nlm.nih.gov/31050513/
  eOn docs mention native GP-accelerated NEB: https://eondocs.org/user_guide/neb.html

  Что это.
  Вместо считать DFT forces на каждом image на каждой итерации, строится surrogate PES через Gaussian Process.

  Схема:

  немного дорогих DFT E/F
          ↓
  GP surrogate PES
          ↓
  много дешёвых NEB-шагов на GP
          ↓
  новые DFT calls только где uncertainty высокая

  Что нужно на входе.

  - descriptors координат;
  - E/F samples;
  - uncertainty model;
  - active learning loop.

  Для нас полезно как pre-flight?
  Да, но осторожно.

  Полезно:

  - можно оценивать uncertainty path-а;
  - можно делать cheap "is this path learnable / smooth?" screen;
  - наш локальный per-image surrogate — маленький локальный вариант этой идеи.

  Опасно:

  - Fe-S cubane, spin states, Fe-H hydride, U/magnetism — всё это OOD-risk;
  - GP может красиво интерполировать физически неверную поверхность;
  - для наших минералов "низкая surrogate energy" не равна "DFT-valid path".

  Вывод:
  GP-NEB хорош как uncertainty-aware preflight layer, но не как oracle. Для нас правильнее:
  MACE + CHGNet + structural/PH/magnetic gates + DFT singlepoints, а GP — как дополнительная локальная память/uncertainty, не как источник
  истины.

  ———

  # Как это связано с нашей целью

  Цель:

  > проверить input перед DFT, чтобы не делать пустые прогоны; решить максимально математически/топологически, а не брутфорсом.

  Тогда эти методы надо использовать не как "конкурентов", а как категории сигналов.

  ## Что нам реально нужно

  Перед DFT NEB надо ответить:

  1. Endpoint integrity: A и B разные basins или same-basin artifact?
  2. Path topology: между ними есть осмысленный mountain pass, а не multi-site/multi-saddle mess?
  3. Path stability: cheap path не скатывается в нефизичную яму?
  4. Magnetic sheet consistency: endpoints/images на одном spin sheet?
  5. Calculator agreement: MACE и CHGNet хотя бы качественно согласны?
  6. Parameter robustness: verdict не меняется от k_spring, image count, reparam, optimizer?

  Вот где prior art помогает.

  ## Как это встраивается в наш framework

  L0  pristine / symmetry / chemistry gates
  L3  topological/geometric path diagnostics:
      - PH
      - internal-coordinate same-basin score
      - dense-spline barrier
      - plain string
      - dyNEB / standard NEB
      - adaptive reparam / OM-like spacing
      - optional SPM local saddle probe
  L5  only then DFT NEB

  Key principle:

  Если разные дешёвые математические взгляды дают разные ответы,
  это не "надо запустить DFT и посмотреть".
  Это INVESTIGATE / NO-GO.

  ———

  # Практический вывод

  Самый сильный путь сейчас:

  1. Не пытаться "победить NEB literature".
  2. Сделать pre-flight theorem-like certificate:
      - endpoints distinct;
      - path not same-basin;
      - no spin sheet discontinuity;
      - MLIP cross-check не OOD;
      - barrier/path qualitative stable under method perturbations.

  3. Использовать SPM/dyNEB/OM/OCINEB/GP-NEB как диагностические perturbations, а не как замену QE/ABACUS/jDFTx.

  То, что добавлено в NEB-AGM — same_basin_gate — ровно из этой философии:
  лучше остановиться с диагнозом "input бессмысленный", чем получить красивый нулевой DFT barrier за деньги.

  ———

  ## Фокус на магнетизм

  Отдельный больной вопрос: почему хочется больше алгоритмически решить эту задачу — из-за магнетизма.
  У MACE/CHGNet совсем с этим плохо, у остальных хоть и есть всякие +U, и Umbrella, но они тоже пока себя не показали на наших FeS минералах,
  поэтому фокус — на магнетизм при рассмотрении гипотез для pre-flight.

  Pre-flight нужно сфокусировать именно вокруг магнитной неоднозначности: что можно проверить до DFT NEB, чему нельзя доверять у MLIP, и где
  нужен минимальный DFT, чтобы не строить путь между разными spin-листами. (Детали — в `MAGNETIC_FIRST_PREFLIGHT_PLAN.md`.)
