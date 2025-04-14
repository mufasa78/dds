from data.translations import TRANSLATIONS

# Available languages
LANGUAGES = {
    'en': {
        'name': 'English',
        'native_name': 'English'
    },
    'zh': {
        'name': 'Chinese',
        'native_name': '中文'
    }
}

def get_translation(language_code):
    """
    Get a translation function for the specified language
    
    Args:
        language_code (str): Language code (e.g., 'en', 'zh')
        
    Returns:
        callable: A translation function
    """
    # Default to English if the requested language doesn't exist
    if language_code not in LANGUAGES:
        print(f"Warning: Language '{language_code}' not found, falling back to English")
        language_code = 'en'
        
    def translate(key):
        """
        Translate a key to the specified language
        
        Args:
            key (str): Translation key
            
        Returns:
            str: Translated text
        """
        # Check if the key exists in our translations dictionary
        if key in TRANSLATIONS:
            # Check if the requested language exists for this key
            if language_code in TRANSLATIONS[key]:
                return TRANSLATIONS[key][language_code]
            # Fall back to English if the language doesn't exist
            elif 'en' in TRANSLATIONS[key]:
                print(f"Warning: No {language_code} translation for key '{key}', falling back to English")
                return TRANSLATIONS[key]['en']
        
        # If the key doesn't exist, return the key itself as a fallback
        print(f"Warning: No translation found for key '{key}'")
        return key
    
    return translate