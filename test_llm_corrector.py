# test_llm_corrector.py
# (Using the OpenAI-compatible client)

import os
import sys
from openai import OpenAI  # <-- Use the official openai library
from dotenv import load_dotenv

# --- System prompt remains the same ---
CORRECTION_SYSTEM_PROMPT = """
You are an expert Sinhala editor and proofreader with a strong background in Political Science.
Your task is to correct a raw, real-time transcription of spoken Sinhala,
which is on the topic of Political Science.

Follow these rules strictly:
1.  **Correct Spelling:** Fix any Sinhala spelling mistakes (e.g., "වරදි" -> "වැරදි").

2.  **Fix Pronunciation Errors:** The ASR may write a phonetically similar but incorrect word.
    Change it to the nearest correct word that makes sense in the context.
    (e.g., ASR: "විශිය" -> Correct: "විෂය", "ඉස්සුන්ගේ"-> "මිනිස්සුන්ගේ","අපගී"->"අප ගේ", "ඉතිහාසී සිට"-> "ඉතිහාසයේ සිට").

3.  **Contextual Word Correction:** The ASR may produce words that are phonetically plausible
    but are invalid or meaningless in the context (e.g., "ආක්පිතියි").
    First, analyze the entire speech flow to understand the topic. Then,
    use your **Political Science knowledge** to change these words to the
    *most logical and meaningful word* that fits the context.
    (e.g., "රාජ්‍ය ආණ්ඩු ක්‍රම ආක්පිතියි." -> "රාජ්‍ය ආණ්ඩු ක්‍රම ආකෘතියකි.")

4.  **Remove Filler Words:** Remove all filler words, "dumb text," and stutters
    (e.g., "අ...", "ම්ම්...", "ආ...", "හ්ම්", "ත්ත්ත්", "ත්ත්", etc.).

5.  **Fix Grammar:** Correct basic grammatical errors to ensure the text is readable.

6.  **Preserve Meaning:** Do NOT add new information or change the original speaker's intended meaning.
    Your goal is to clarify, not to rewrite. (eg., "සෞඛි සේවා"->"සෞඛ්‍ය සේවා", "සේභාවන්"->"සේවාවන්","නැතම්"->"නැත්තම්")

7.  **RESPONSE:** Respond ONLY with the corrected, clean Sinhala text.
    Do not add any pre-amble, explanation, or chat text like "Here is the correction:".
    Just give the text.
"""

# Model name remains the same
MODEL_NAME = "mistralai/mistral-7b-instruct:free"


def get_openai_client():
    """Initializes and returns the OpenAI client pointed at OpenRouter."""

    # --- NEW: Load variables from .env file ---
    load_dotenv()
    # ----------------------------------------
    api_key = os.environ.get("OPENROUTER_API_KEY")
    print(api_key)
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        print("Please set the variable and try again.")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",  # <-- Point to OpenRouter
        api_key=api_key,
    )
    return client


def correct_sinhala_text(client, raw_text):
    """
    Sends the raw text to the LLM for correction.
    """
    if not raw_text or not raw_text.strip():
        print("Skipping empty text.")
        return ""

    print(f"--- Sending to LLM ---")
    print(f"RAW:        {raw_text}")

    try:
        completion = client.chat.completions.create(
            # Optional headers for OpenRouter analytics
            extra_headers={
                "HTTP-Referer": "http://localhost/sinhala-asr",  # Optional
                "X-Title": "Sinhala ASR Corrector"  # Optional
            },
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": CORRECTION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": raw_text,  # <-- Simple text content
                },
            ],
        )

        corrected_text = completion.choices[0].message.content.strip()
        print(f"CORRECTED:  {corrected_text}\n")
        return corrected_text

    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        return raw_text


# --- Main test block (no change) ---
if __name__ == "__main__":

    print("Initializing LLM Corrector Test (using OpenAI client)...\n")

    # 1. Initialize the client
    client = get_openai_client()

    # 2. Define your test cases
    test_cases = [
        "අද නොම ඔබ සමඟ බිදාගන්නේ දීශපාලන විද්‍යාව ලූකික් ලොකේ මූලික පියවා. ඇන් මෙය කිසිම නිශ්චිත විෂයක් නොවේ. එය අපගේ ජීවිතේ සෑම අයියි. ඉතේ සෑම අන්සේයකටම බලපාන සමාජ්‍ය බලවේග, බලධාරිං සතීරණ ගැනීම් වල ඉංහිංවළ රහස් විස්තර විශයක් ඒ පමණක් නොවෙයි. අපගී රටි ඉතිහාසී සිට නූතන ලෝකය දක්වා මිනිස්සුන්ගේ අනාගත් ඉස්සුන්ගේ අනාගතය හැඩ ගස්සන බල වීගිය ත්ත්ත් යොක් ප්‍රතියි. ඔබ කවදා හා සිට්වද ඔබ චන්දයක් තැබීම නොබේ ජීවිතය මග පෙන්වන සම්පත් අධ්‍යාපනය සෞඛි සේවා සෞඛ්‍ය සේභාවන් වෙනස් වෙන්නෙ කිහිත් කොහොමද කියලා. නැතම් ලෝක ලෝක නායකයංගේ තිර්ණය වලි අපගේ දෛනික ජීමිතියට බල බලපෑම් ඇතිවන්නේ කෙහෙම කොහොමද කිය මේ දේශපාලන විද්‍යාව එස් යල්ල පැහැදිරි කරනව. එය රාජ්‍යයන්ගේ බලගැන් වී ප්‍රහදී සිහත් ජාත්‍යනතර සබඳතා සබද තා සහ සමාධි සාධ්‍යාරණත්තේ අධී ගැන ස්වාභයනු ඉසා පලනු ත්ත්ත් යොක් ප්‍රත්ත් ත්ත්ත් යොක් ප්‍රත්ත් ත්ත්ත් යොක් ප්‍රත්ත් ත්ත්ත් යොක් ප්‍රත්ත් අපි මොලිම බලමු දේශ්‍යපාලන විද්‍යාවෙ මූලික සංකල්ප ත්ත්ත් ත්ත්ත් යොක් ප්‍රතියි. "
    ]

    print("--- Starting Correction Tests ---\n")

    # 3. Run the tests
    for i, text in enumerate(test_cases):
        print(f"--- Test Case {i + 1} ---")
        correct_sinhala_text(client, text)

    print("--- All tests complete. ---")
