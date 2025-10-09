#!/usr/bin/env python3
import sys
import os
import openai

def main():
    prompt = sys.stdin.read().strip()
    if not prompt:
        print("Usage: echo 'prompt' | python answer.py")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    openai.api_key = api_key
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512
    )

    print(response.choices[0].message.content.strip())

if __name__ == "__main__":
    main()
