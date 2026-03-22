import os
import time
import requests

MODEL = "swiss-ai/apertus-8b-instruct"


def ask_publicai(
    payload=None,
    prompt=None,
    model=None,
    user_agent=None,
    api_key=None,
    timeout=30,
    max_retries=3,
    backoff_factor=2,
):
    if payload is None and prompt is None:
        raise ValueError("Either payload or prompt must be provided.")

    if not user_agent:
        raise ValueError("User-Agent must be provided.")

    if api_key is None:
        api_key = os.getenv("PUBLICAI_KEY")
        if not api_key:
            raise ValueError("API key not provided and PUBLICAI_KEY not set.")

    if model is None:
        model = MODEL

    url = "https://api.publicai.co/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": user_agent,
    }

    if payload is None:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

    last_exception = None

    for attempt in range(max_retries):
        try:
            start_time = time.time()

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            elapsed = time.time() - start_time
            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout as e:
            last_exception = e
            error_type = "Timeout"

        except requests.exceptions.ConnectionError as e:
            last_exception = e
            error_type = "ConnectionError"

        except requests.exceptions.HTTPError as e:
            # Don't retry 4xx errors (client-side issues)
            if 400 <= response.status_code < 500:
                raise RuntimeError(f"Client error {response.status_code}: {response.text}")
            last_exception = e
            error_type = "HTTPError"

        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Unexpected API response format: {response.text}")

        # Retry logic
        if attempt < max_retries - 1:
            sleep_time = backoff_factor ** attempt
            # print(f"[PublicAI] {error_type}, retrying in {sleep_time}s...")
            time.sleep(sleep_time)
        else:
            break

    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_exception}")