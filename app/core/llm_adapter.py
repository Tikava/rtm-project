"""
LLM adapter. Uses OpenAI (chat) when включено через переменные окружения.
Если LLM_DISABLED или нет ключа, вернёт заглушку с пояснением.
"""
import os
from typing import List, Dict, Any


SYSTEM_PROMPT = "Ты архитектор, кратко описываешь домены данных."
USER_PROMPT_TEMPLATE = (
    "Дана декомпозиция таблиц БД на домены. Напиши по одному предложению на русском для каждого домена: "
    "что он описывает и зачем его выделять. Формат: 'Домен N: текст'.\nДомены:\n{domain_lines}"
)


def llm_enabled() -> bool:
    return os.getenv("LLM_ENABLED", "0") == "1"


def _get_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None, "Библиотека openai не установлена (pip install openai)."
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY не задан."
    base_url = os.getenv("OPENAI_BASE_URL")  # опционально для прокси/azure
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, None


def summarize_domains_with_llm(domains: List[List[str]], summaries: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
    {
      "enabled": bool,
      "provider": str,
      "items": [ { "tables": [...], "summary": "..." }, ... ],
      "note": str
    }
    """
    if not llm_enabled():
        return {
            "enabled": False,
            "provider": None,
            "items": [],
            "note": "LLM выключена (установите LLM_ENABLED=1, задайте OPENAI_API_KEY)."
        }

    client, err = _get_openai_client()
    if err:
        return {
            "enabled": False,
            "provider": None,
            "items": [],
            "note": err
        }

    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    # Формируем краткий промпт
    domain_lines = []
    for idx, comp in enumerate(domains):
        label = summaries[idx].get("label") if idx < len(summaries) else ""
        domain_lines.append(f"{idx+1}. {', '.join(comp)} (подсказка: {label})")

    prompt = USER_PROMPT_TEMPLATE.format(domain_lines="\n".join(domain_lines))

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        content = resp.choices[0].message.content if resp.choices else ""
    except Exception as e:
        return {
            "enabled": False,
            "provider": "openai",
            "items": [],
            "note": f"Ошибка обращения к OpenAI: {e}"
        }

    items = []
    lines = content.splitlines()
    for idx, comp in enumerate(domains):
        # ищем строку с "домен {idx+1}"
        summary_text = ""
        for line in lines:
            if str(idx+1) in line:
                summary_text = line
                break
        if not summary_text and lines:
            summary_text = lines[min(idx, len(lines)-1)]
        if not summary_text:
            summary_text = f"Домен {idx+1}: {', '.join(comp)}."
        items.append({"tables": comp, "summary": summary_text})

    return {
        "enabled": True,
        "provider": "openai",
        "items": items,
        "note": f"Модель {model}"
    }
