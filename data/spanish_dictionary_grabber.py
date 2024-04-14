from wiktionaryparser import WiktionaryParser

# Assuming you already have a list of nouns
'''
nouns = [
    "perro", "gato", "casa", "árbol", "coche", "niño", "niña", "manzana", "computadora", "teléfono",
    "mesa", "silla", "libro", "pelota", "juego", "amigo", "amiga", "ciudad", "calle", "trabajo",
    "comida", "agua", "ropa", "música", "familia", "hogar", "arte", "naturaleza", "sol", "luna",
    "estrella", "océano", "playa", "montaña", "río", "aire", "fuego", "tierra", "día", "noche",
    "mañana", "tarde", "noche", "hora", "minuto", "segundo", "número", "color", "forma", "tamaño",
    "punto", "línea", "forma", "luz", "sombra", "movimiento", "cambio", "acción", "palabra", "idea",
    "sentimiento", "emoción", "amor", "odio", "felicidad", "tristeza", "sorpresa", "miedo", "paz",
    "guerra", "libertad", "justicia", "verdad", "mentira", "amistad", "relación", "historia", "cultura",
    "sociedad", "gobierno", "política", "economía", "educación", "tecnología", "ciencia", "medicina",
    "naturaleza", "animal", "planta", "insecto", "pez", "ave"
]
'''
nouns = [
    "perro", "casa", "árbol", "carro"
]

def get_spanish_definitions(word, language='spanish'):
    try:
        print(f"Fetching definition for '{word}'...")
        parser = WiktionaryParser(lang_code='es')
        parser.set_default_language(language)
        definitions = parser.fetch(word, language)
        print(definitions)
        if definitions:
            return [definition['definitions'][0]['text'] for definition in definitions]
        else:
            return []
    except Exception as e:
        print(f"Error fetching definition for '{word}': {e}")
        return []

# Fetch definitions for each noun
noun_definitions = {}
for noun in nouns:
    definitions = get_spanish_definitions(noun)
    noun_definitions[noun] = definitions

# Print the noun along with its definitions
for noun, definitions in noun_definitions.items():
    print(f"{noun}: {definitions}")
