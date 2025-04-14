from data.translations import TRANSLATIONS

# Dictionary of supported languages
LANGUAGES = {
    "en": {"english": "English", "native": "English"},
    "zh": {"english": "Chinese", "native": "中文"}
}

def get_translation(language_code):
    """
    Get a translation function for the specified language
    
    Args:
        language_code (str): Language code (e.g., 'en', 'zh')
        
    Returns:
        callable: A translation function
    """
    # Default to English if language code not supported
    if language_code not in LANGUAGES:
        language_code = 'en'
    
    def translate(key):
        """
        Translate a key to the specified language
        
        Args:
            key (str): Translation key
            
        Returns:
            str: Translated text
        """
        # Get translations for the language
        translations = TRANSLATIONS.get(language_code, {})
        
        # Return the translation or the key if not found
        return translations.get(key, TRANSLATIONS.get('en', {}).get(key, key))
    
    return translate
