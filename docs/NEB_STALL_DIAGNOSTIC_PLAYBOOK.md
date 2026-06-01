# NEB-Stall Diagnostic Playbook

**Что:** воспроизводимая процедура «DFT NEB застрял — почему и как чинить». Отделяет три разных корня (оптимизатор/пружина · неправильные эндпоинты/multi-site · магнитная неоднозначность) дешёвыми ($0) шагами ДО повторного дорогого DFT.

**Когда применять:** NEB не сходится — fmax встал на floor (≫ target) и/или **разворачивается вверх**, энергия верхнего image заморожена, BFGS/FIRE крутится сотни шагов без падения энергии.

**Валидировано:** pyrite V_Fe (корень = пружина/оптимизатор) и marcasite V_Fe (корень = неправильные эндпоинты + магнетизм). Два противоположных диагноза одной процедурой.

---

## Процедура (4 шага, от дешёвого к дорогому)

### Шаг 1 — Анализ `neb.traj`: где живёт сила? (бесплатно, локально)

Модуль: the `neb_stuck_analysis` gate. Что считает:
- per-image E_rel + fmax (какой image застрял),
- **per-atom force breakdown на worst image** (на каких атомах сидит остаточная сила),
- ближайшие соседи мигрирующего атома.

Интерпретация распределения силы:
| Сила сосредоточена на… | Вероятный корень |
|---|---|
| мигрирующем атоме (H) + якоре (S) → металл < 1% | **геометрия пути / пружина** (НЕ электроника) |
| металле / размазана по решётке | возможна **электроника** (nspin/U/smearing) |

> Урок: на пирите сила была H 52% + S 48%, **Fe 0.8%** → сразу сняло гипотезу про nspin/V_Fe-дырку. Не гадать — измерять.

### Шаг 2 — Сравнить стартовый IDPP-путь с финальным band

Модуль: the `neb_path_geom` gate (читает первые `n_movable` кадры traj = IDPP, последние = финал).
- **IDPP чистый, финал — каша БЕЗ падения энергии** → band «скатился с гребня» = оптимизатор/пружина. Идти в Шаг 4 → ожидать чистый MLIP.
- **IDPP уже плохой** (мигрирующий атом грязит соседей) → проблема пути/эндпоинтов изначально.

### Шаг 3 — Эндпоинты: симметрия + магнитное состояние

- **Геометрия:** non-H смещение endA→endB (MIC, без relabel или Hungarian, см. the `symmetry_preflight_general` gate). <0.15 Å → почти зеркальная пара; большое → разные сайты.
- **Магнетизм (если nspin=2):** `grep "total magnetization\|absolute magnetization\|^!" sp_end{A,B}/espresso.pwo`. **Разные total/abs mag на эндпоинтах = magnetic-state mismatch (condition C2)** → NEB интерполирует смену спина → не сойдётся. Проверить starting_magnetization + наличие +U.

### Шаг 4 — $0 MLIP-валидация: MACE + CHGNet NEB на ТЕХ ЖЕ DFT-эндпоинтах

Модуль: the `neb_vfe_validate_mlip` runner (env `MLIP_MODEL=mace|chgnet`, `OUT_TAG`, `MINERAL_NAME`). Протокол: IDPP(mic) → plain NEB (k=1.0) → CI-NEB, FIRE.
Для нового минерала — извлечь эндпоинты (the `extract_endpoints` helper как шаблон) и адаптировать `OUT_TAG`.

Зачем именно MLIP: дёшево ($0, ~1–3 мин на локальном GPU), и **MACE/CHGNet не моделируют спин** → отделяют геометрию пути от магнетизма. Кросс-чек двух MLIP ловит OOD-артефакты одной (как pent MACE 30 эВ).

---

## Таблица интерпретации Шага 4

| MLIP-результат | Диагноз | Что делать |
|---|---|---|
| Обе сходятся чисто, барьер физичный, ΔE_endpoints как DFT, эндпоинты = минимумы | **Путь ОК, корень — пружина/оптимизатор** | k_spring ×5–10 (0.3→1.5–3.0), plain→CI, `NEBOptimizer(ode)`/LBFGS вместо FIRE. (= pyrite) |
| Обе дают band-collapse (промежуточные ниже эндпоинтов), barrier_fwd=0 | **Эндпоинты НЕ истинные минимумы / multi-site** | L2 multi-endpoint enumeration → найти реальные минимумы → перевыбрать пару. (= marcasite) |
| Обе воспроизводят асимметрию ΔE_endpoints (spin-free!) | **Асимметрия геометрическая**, не магнитная | разные сайты — это реальность, не артефакт спина |
| ΔE_endpoints(MLIP)≈0, но DFT≫0 + разные mag (Шаг 3) | **Асимметрия магнитная** (C2) | одинаковый starting_mag на всех images; +U; (spin-aware MLIP — research) |
| MACE и CHGNet СИЛЬНО расходятся (×10+) | **OOD-артефакт одной MLIP** | доверять согласию, не магнитуде; L4 DFT single-point |

---

## Два разобранных кейса (эталоны)

### Pyrite V_Fe — корень: пружина/оптимизатор ✅ решено
- Шаг 1: сила H 52% + S 48%, Fe 0.8% → не электроника.
- Шаг 2: IDPP идеален (симметричный hop, 0.628 эВ), финал — каша без падения E → ridge-rolling.
- Шаг 4: MACE **182 meV**, CHGNet **223 meV** — обе сходятся за 29–68 шагов, согласие, в предсказанном 150–400 meV.
- **Fix:** k 0.3→1.5–3.0 + plain→CI. Ожидаемый DFT-барьер ~200–300 meV.

### Marcasite V_Fe — корень: неправильные эндпоинты + magnetic mismatch
- Шаг 3: endA/endB разные mag (1.67/2.56 vs 1.13/1.91 μB), non-H disp 0.13 Å (решётки идентичны), но ΔE=174 meV.
- Шаг 4: MACE −198 / CHGNet −103 meV (обе spin-free воспроизводят асимметрию → она ГЕОМЕТРИЧЕСКАЯ); обе дают band-collapse (image7 −601/−364 ниже endB) → эндпоинты не минимумы.
- **Fix:** L2 multi-endpoint enumeration → перевыбор пары + magnetic consistency (+U=2). НЕ рецепт пирита.

---

## Реестр модулей

| Модуль | Назначение |
|--------|------------|
| the `neb_stuck_analysis` gate | Шаг 1 — per-image E/fmax + per-atom force breakdown из neb.traj |
| the `neb_path_geom` gate | Шаг 2 — IDPP vs финальный band, геометрия мигрирующего атома |
| the `extract_endpoints` helper | извлечь endA/endB из QE `.pwi`/`.pwo` + сравнить геометрию (шаблон для нового минерала) |
| the `neb_vfe_validate_mlip` runner | Шаг 4 — MACE/CHGNet NEB на DFT-эндпоинтах (параметризован `OUT_TAG`/`MINERAL_NAME`/`MLIP_MODEL`) |

**Связано:** the universal "NEB band rolls off ridge" lesson, `EVIDENCE_FRAMEWORK_V2_COMPLETE.md` (L0–L6), the same-basin endpoint lesson (другой failure mode — same-basin).

---

*Создан по результатам диагностики pyrite + marcasite V_Fe NEB.*
