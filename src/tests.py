import unittest
from get_article import get_article
from cognate_analysis import cognate_analysis, get_cognate, sentence_to_word_list


class MyTestCase(unittest.TestCase):
    def test_article_word_count(self):
        # Call the get_article function to get an article
        article_text = get_article(language_code="es")

        # Calculate the word count of the article
        word_count = len(article_text.split())

        print("Article is", article_text)

        # Check if the word count is in the range 10-50
        self.assertTrue(10 <= word_count <= 50, f"Word count {word_count} is not in the range 10-30")

    def test_article_cleaner(self):
        # Call the get_article function to get an article
        article_text = get_article(language_code="es")
        print("Article is", article_text)

        self.assertTrue(('[' not in article_text) and ('('  not in article_text))


    def test_sentence_cleaner(self):
        text = "La universidAd. el cabAllo; !!! ; ; ;es muy fáCil"
        expected_cleaned_word_list = ['la', 'universidad', 'el', 'caballo', 'es', 'muy', 'fácil']
        actual_cleaned_word_list = sentence_to_word_list(text)
        self.assertTrue(actual_cleaned_word_list == expected_cleaned_word_list)

        text = "La universidad, el caballo, la computadora, general, universidad, caballo"
        expected_cleaned_word_list = ['la', 'universidad', 'el', 'caballo', 'la', 'computadora', 'general', 'universidad', 'caballo']
        actual_cleaned_word_list = sentence_to_word_list(text)
        self.assertTrue(actual_cleaned_word_list == expected_cleaned_word_list)

    def test_cognate_generator(self):
        # Call the get_article function to get an article
        text = "La universidad, el caballo, la computadora, general, universidad, caballo"
        word_list = sentence_to_word_list(text, trim_small_words=True)
        cognates, non_cognates, ratio = cognate_analysis(word_list)
        print("cognates = ", cognates, " non_cognates = ", non_cognates)
        self.assertTrue(cognates == {'universidad': 'university', 'computadora': 'computer', 'general': 'general'} and non_cognates == {'caballo': 'horse'})

        text = "doctor, programador, tenemos, hablamos"
        word_list = sentence_to_word_list(text, trim_small_words=True)
        cognates, non_cognates, ratio = cognate_analysis(word_list)
        print("cognates = ", cognates, " non_cognates = ", list(non_cognates.keys()))
        self.assertTrue(cognates == {'doctor': 'doctor', 'programador': 'programmer'}
                        and ("tenemos" in list(non_cognates.keys()))
                        and ("hablamos" in list(non_cognates.keys())))

    def test_has_cognate(self):
        words = ["universidad", "profesor", "computadora", "alcalde", "aceite"]
        expected_cognate_array = [True, False, True, False, False]
        actual_cognate_array = [bool(get_cognate(i)) for i in words]
        print(actual_cognate_array)
        self.assertTrue(expected_cognate_array == actual_cognate_array)

if __name__ == '__main__':
    unittest.main()