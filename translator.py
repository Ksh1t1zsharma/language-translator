from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate(text, source_lang, target_lang):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translated_text

def language_menu():
    languages = {
        "en": "English",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "hi": "Hindi",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "ar": "Arabic"
    }

    print("Available Languages:")
    for code, name in languages.items():
        print(f"{code} : {name}")
    
    return languages

if __name__ == "__main__":
    languages = language_menu()

    source_lang = input("Enter source language code (example 'en' for English): ").strip()
    target_lang = input("Enter target language code (example 'fr' for French): ").strip()

    if source_lang not in languages or target_lang not in languages:
        print("Invalid language codes entered. Please try again.")
    else:
        text = input(f"Enter text in {languages[source_lang]}: ")
        translation = translate(text, source_lang, target_lang)
        print(f"\nTranslated text in {languages[target_lang]}: {translation}")
