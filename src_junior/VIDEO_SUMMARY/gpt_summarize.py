"""Text summarization using ChatGPT"""
from openai import OpenAI
import httpx

OPENAI_API_KEY="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"

def summary_prompt(input_text: str) -> str:
    """
    Build prompt using input text of the video.
    """
    prompt =\
    f"""
    Привет! Приведи, пожалуйста, краткое содержание текста, отразив только ключевую информацию.
    Объем содержания - не более 50 слов.
    Входной текст: {input_text}
    """
    return prompt


def summarize_text(input_text: str) -> str:
    """
    Summarize input text of the video.

    Examples
    --------
    >>> summary = summarize_text(video_text)
    >>> print(summary)
    'This video explains...'
    """
    # generate the prompt
    prompt = summary_prompt(input_text)
    # make a request to GPT
    client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client
                    (proxies="socks5://CbL1M6:ECgv5Q@196.18.14.223:8000"))
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.7)

    ### output the answer
    return completion.choices[0].message.content

input_text = """Всем привет! Команда Яндекс объявляет начало весеннего отбора
                стажеров на обучение в сегменте науки данных, машинного обучения
                и обработки естественного языка.
                Начала конкурса - 19 февраля. Ждем всем, желающих протестировать свои
                hard-skills и испытать свою нервную систему.
                С любовью, Яндекс!"""
print(summarize_text(input_text))