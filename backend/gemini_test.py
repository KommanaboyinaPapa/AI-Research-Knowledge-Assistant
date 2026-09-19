import os

from google import genai


def main() -> None:
    # Read the API key from the environment instead of storing it in the source code.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")

    # Create a Gemini client using the API key.
    client = genai.Client(api_key=api_key)

    prompt = "Explain deep learning in simple terms in 3 sentences."

    # Send the prompt to Gemini using the Interactions API.
    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    print(interaction.output_text)


if __name__ == "__main__":
    main()
