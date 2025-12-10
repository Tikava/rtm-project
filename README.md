# AI-Assisted Migration Prototype

Прототип сервиса на FastAPI, который принимает схему реляционной БД и помогает разложить её на домены, сравнить базовый и эвристический анализ, а также выдать черновой план миграции (сервисы, владение таблицами, план шардирования).

Суть сервиса:
- Принимает схему БД (tables, relations, usage), валидирует и нормализует.
- Строит граф, считает baseline и расширенный “AI” разбор (эвристики + доменные правила).
- Возвращает метрики связности, объяснения доменов (включая LLM-описания) и сравнение baseline vs AI, а также timing и token_usage.
- Генерирует черновой план миграции: границы сервисов, карта владения таблицами и план шардирования.

## Быстрый старт
- Установите зависимости: `pip install -r requirements.txt`
- Запустите сервер: `uvicorn main:app --reload`
- Откройте docs: `http://127.0.0.1:8000/docs`

## Основные эндпоинты
- `GET /health` — проверка живости.
- `POST /analysis` — принимает JSON-схему (`tables`, `relations`, `usage`), возвращает baseline и AI-разбиение, метрики и сравнение.
- `POST /suggestions` — принимает результат `/analysis` и отдаёт черновой план миграции с границами сервисов и картой владения.
- В ответе `/analysis` дополнительно: `timings_ms` (время baseline и AI), `token_usage` (prompt/completion/total при включённой LLM).

## Пример запроса
```bash
curl -X POST http://127.0.0.1:8000/analysis \
  -H "Content-Type: application/json" \
  -d @app/data/example_schema.json
```

## Что внутри
- `main.py` — FastAPI приложение и роуты.
- `app/core/ai_engine.py` — графовый baseline, эвристический “AI”, метрики и сравнение, публичная функция `analyze_schema`.
- `app/core/llm_adapter.py` — описание доменов через LLM (OpenAI), включение `LLM_ENABLED=1` и `OPENAI_API_KEY`.
- `app/utils/db_tools.py` — нормализация входной JSON-схемы.
- `app/services/migration_service.py` — генерация плана миграции на основе AI-доменов.
- `app/data/example_schema.json` — образец схемы для теста.

## Алгоритм (формально)
1. Нормализация/валидация схемы (Pydantic).
2. Построение графа FK и co-usage, вычисление порога usage (медиана).
3. Baseline: компоненты связности графа.
4. AI: разрез слабых связей по usage, семена доменов (identity/auth, catalog/inventory, orders/checkout, payments, promo, reference), доменные правила (identity/audit/payments/reference вынесены), приклейка мелких компонентов, ограниченное слияние по сильной ко-используемости.
5. Пост-обработка join/utility таблиц.
6. Метрики: domain_count, avg_internal_coupling, cross_domain_edges, over_segmentation, ownership_conflicts. Domain graph: cross-domain рёбра/веса.
7. Лейблинг: эвристические summaries + LLM-описания (опционально).

## Идеи для развития
- Добавить настоящую ML-модель или LLM-подсказки для разбиения.
- Парсить SQL-дампы в схему, а не только JSON.
- Улучшить метрики и визуализацию доменов.
- Покрыть API автотестами и добавить CLI утилиту для анализа.
- Визуализации: heatmap cross-domain usage, граф связей, baseline vs AI разбиение.
- Case-study таблица: ошибки кластеризации/случаи, где baseline лучше AI.
- Benchmark: сравнение baseline (benchmark) vs AI-engine (tested) на наборах схем, метрики и дельты.

## Настройка LLM (OpenAI)
- Задайте `LLM_ENABLED=1` и `OPENAI_API_KEY=...`.
- Опционально: `OPENAI_MODEL` (по умолчанию gpt-4o-mini), `OPENAI_BASE_URL` для кастомного/прокси эндпоинта.
- Если переменные не заданы или возникнет ошибка запроса, сервис вернёт заглушку и пояснение в поле `llm_summaries.note`.

## Платформа/окружение
- Python 3.x, FastAPI + uvicorn, кастомный AI-engine на графовых эвристиках.
- Опционально: OpenAI для LLM-лейблинга (token_usage в ответе).
- Тестовый пример: схема e-commerce (~18 таблиц, FK + usage).
