"""
Tests for _extract_key_facts_llm: routing, JSON parsing, fallbacks.
"""
import sys, json
sys.path.insert(0, '.')

from backend.llm_expert import LLMExpert

sample_text = (
    'Инфляция в 2024 году составила 7.4%. '
    'ВВП вырос на 3.6 трлн руб. '
    'Прогноз ставки ЦБ снижение до 18%.'
)

def make_expert_with_mock(mock_response: str):
    """Return LLMExpert with a stub client and mocked _call_yandex_gpt."""
    expert = LLMExpert.__new__(LLMExpert)
    # Minimal attribute init
    expert.api_key = 'fake'
    expert.folder_id = 'fake'
    expert.model = 'yandexgpt-lite'
    expert.base_url = ''

    class FakeClient:
        pass
    expert.client = FakeClient()  # truthy

    expert._call_yandex_gpt = lambda prompt, instructions='': mock_response
    return expert


def test_no_client_regex_fallback():
    expert = LLMExpert()  # no API keys → client = None
    facts = expert._extract_key_facts_llm(sample_text, 'http://test.com')
    assert len(facts) > 0, 'Should fall back to regex'
    print('✅ Test 1 (no client → regex fallback):', facts)


def test_good_json_response():
    payload = json.dumps({'facts': [
        'Инфляция составила 7.4% в 2024 году',
        'ВВП вырос на 3.6 трлн руб',
        'Ставка ЦБ снижена до 18%',
    ]})
    expert = make_expert_with_mock(payload)
    facts = expert._extract_key_facts_llm(sample_text, 'http://test.com')
    assert facts == [
        'Инфляция составила 7.4% в 2024 году',
        'ВВП вырос на 3.6 трлн руб',
        'Ставка ЦБ снижена до 18%',
    ], f'Unexpected: {facts}'
    print('✅ Test 2 (good JSON):', facts)


def test_markdown_wrapped_json():
    inner = json.dumps({'facts': ['Инфляция 7.4%', 'ВВП +3.6 трлн']})
    wrapped = '```json\n' + inner + '\n```'
    expert = make_expert_with_mock(wrapped)
    facts = expert._extract_key_facts_llm(sample_text, 'http://test.com')
    assert 'Инфляция 7.4%' in facts, f'Unexpected: {facts}'
    print('✅ Test 3 (markdown JSON):', facts)


def test_json_embedded_in_prose():
    inner = json.dumps({'facts': ['Дефицит кадров достиг 1 млн человек']})
    prose = f'Вот результат анализа: {inner} Конец анализа.'
    expert = make_expert_with_mock(prose)
    facts = expert._extract_key_facts_llm(sample_text, 'http://test.com')
    assert 'Дефицит кадров достиг 1 млн человек' in facts, f'Unexpected: {facts}'
    print('✅ Test 4 (JSON embedded in prose):', facts)


def test_broken_json_falls_back_to_regex():
    expert = make_expert_with_mock('Это совсем не JSON!')
    facts = expert._extract_key_facts_llm(sample_text, 'http://test.com')
    assert len(facts) > 0, 'Regex fallback should return something'
    print('✅ Test 5 (broken JSON → regex fallback):', facts)


def test_empty_response_falls_back_to_regex():
    expert = make_expert_with_mock('')
    facts = expert._extract_key_facts_llm(sample_text, 'http://test.com')
    assert len(facts) > 0, 'Regex fallback should return something'
    print('✅ Test 6 (empty response → regex fallback):', facts)


def test_max_facts_respected():
    payload = json.dumps({'facts': [f'Факт {i}' for i in range(20)]})
    expert = make_expert_with_mock(payload)
    facts = expert._extract_key_facts_llm(sample_text, 'http://test.com', max_facts=7)
    assert len(facts) <= 7, f'Expected ≤7 facts, got {len(facts)}'
    print(f'✅ Test 7 (max_facts respected): got {len(facts)} facts')


def test_legacy_alias():
    """_extract_key_facts should still delegate to regex when client is None."""
    expert = LLMExpert()
    facts_alias = expert._extract_key_facts(sample_text, max_facts=5)
    facts_direct = expert._extract_key_facts_regex(sample_text, max_facts=5)
    assert facts_alias == facts_direct, 'Legacy alias must match regex output'
    print('✅ Test 8 (legacy alias _extract_key_facts):', facts_alias)


if __name__ == '__main__':
    print('=' * 60)
    print('LLM Fact Extraction Tests')
    print('=' * 60)
    test_no_client_regex_fallback()
    test_good_json_response()
    test_markdown_wrapped_json()
    test_json_embedded_in_prose()
    test_broken_json_falls_back_to_regex()
    test_empty_response_falls_back_to_regex()
    test_max_facts_respected()
    test_legacy_alias()
    print()
    print('ALL TESTS PASSED ✅')
