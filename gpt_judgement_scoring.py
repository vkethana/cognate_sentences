# TASKS
# 1 - Generate a bunch of sentences in French
# 2 - Catalog the number of cognates in each sentence
# 3 - Figure out correlation between each cognate and the average score it receives

# 1 - Generate french sentences

from backend import get_sentence_starter, make_sentence_object, gpt_scored_rubric_batch, gpt_scored_rubric
from time import sleep
'''
lst = [
    "Le célèbre théâtre de Paris est situé à quelques mètres de la Seine.",
    "Le week-end, beaucoup de touristes visitent Paris pour admirer la Tour Eiffel et le Louvre.",
    "Je recommande le concert de Beyoncé à Paris parce que c'est fantastique et très populaire!",
    "Le musée moderne a une exposition spéciale à quelques mètres de la grande entrée.",
    "L'architecte célèbre a présenté un projet innovant pour le musée moderne.",
    "Le cours d'éducation à Paris est une opportunité unique pour explorer la culture et la langue.",
    "Le chef prépare un steak exquis pour le repas.",
    "Le musicien célèbre joue une symphonie pendant la tempête.",
    "Le festival international célèbre la poésie et la musique contemporaine.",
    "Le menu propose des options variées comme des frites, des pizzas, et des tiramisu.",
    "Le restaurant populaire dans la région offre des menus exceptionnels avec des plates délicieux."
]
'''

'''
lst = [
    "Le célèbre théâtre de Paris est situé à quelques mètres de la Seine.", # 2
    "Je recommande le concert de Beyoncé à Paris parce que c'est fantastique et très populaire!", # 2
    "Le musicien célèbre joue une symphonie pendant la tempête.",
    "Le festival international célèbre la poésie et la musique contemporaine.",
    "Ils coordonnent leurs efforts au sein d'une communauté collaborative, sans dirigeant",
    "Le concile de Constance met fin au grand schisme d'Occident."
]
'''


'''
lst = [
    "Ce n'est pas très différent",
    "Tu veux aller manger avec moi?"
]

scores = []

for i in lst:
    score = gpt_scored_rubric(i)
    scores.append(score)
    print("Score for sentence: ", i, " is: ", score)
    sleep(2)
'''
lst = []

while True:
    # Open JSON file gpt_judgements_for_cognates.json
    # Write the following columns: sentence, cognate, score
    s = make_sentence_object(get_sentence_starter(1.0))
    sentence = s.sentence
    score = gpt_scored_rubric(sentence)
    cognates = s.cognates
    lst.append(s)

    with open("gpt_judgements_for_cognates.json", "a") as f:
        f.write(f'{{"sentence": "{sentence}", "cognates": {cognates}, "score": {score}}},\n')
        print(f'{{"sentence": "{sentence}", "cognates": {cognates}, "score": {score}}},\n')
