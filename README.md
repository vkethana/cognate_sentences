# Motivation
> “Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain.”

Even if you don't speak French, you can probably understand, or at least get the gist of, the above sentence: the French president Emmanuel Macron is assuring the “peuple canadien” (Canadian people) about something involving the “gouvernment français” (French government). 
Imagine reading thousands of sentences like this and gradually acquiring French through cognates you already know. 
This is a type of comprehensible input, a language learning technique popularized by linguist Stephen Krashen in the 1980s. 

Comprehensible input is a good language learning method, but creating it is very hard. 
A native speaker has to painstakingly create thousands of sentences that speakers of another language can understand, gradually increasing the difficulty as the learner progresses. 
Resources like this actually do exist for a number of languages. 
For example, learners of Latin can use *Lingua Latina per se Illustrata*, a book that teaches you Latin using exclusively sentences in Latin. 
However, writing this text took years of effort on the part of Hans Ørberg, a linguist who dedicated a large part of his life to teaching Latin.
Ørberg carefully wrote the sentences in his book to only use cognates an English speaker could understand and to avoid any complicated syntax an English speaker would have a hard time understanding. 

What if there was a way to automate the process of generating "cognateful" sentences using large language models? 
Language models like GPT-4o and o1 have good enough reasoning ability to tell what words in a foreign language are and aren't cognate with English. 
They can reliably generate sentences in other languages like French without any additional training.
In short, this is an ideal use case for LLMs.

# What I did
I made a simple interactive language-learning app, hosted at [cognateful.vkethana.com](https://cognateful.vkethana.com). It teaches you French using stories written exclusively in French.

Users are provided with a brief, ten-sentence story consisting of French sentences. The user's task is to read the sentences and translate them into English. 
The app tells you whether or not your translation is correct using GPT-4o-powered scoring. 
Based on your performance on the exercises, it assigns you a "difficulty" score, which goes up and down depending on your performance. 
Users then are served sentences at appropriate levels of difficulty based on their performance. For those who are curious, the source code for this project can be found [here](https://github.com/vkethana/cognate_sentences).

Caveats: This is just a minimum viable product. 
The number of sentences in the app is limited. 
I don't speak French (yet), so sentences may contain mistakes.
But the interface, scoring system, and sentence generation are all functional, and I think they will work at scale. 
The biggest hurdle to improving the app is increasing the number of sentences while not compromising on sentence quality.

# How I generated and scored the sentences
## Scoring
I have to explain sentence scoring before sentence generation because the scoring system influenced the prompt used to generate the sentences. 
I give o1-preview a sentence and ask it to score its difficulty on a scale of 0 to 3, 0 being very hard to understand for a monolingual English speaker and 3 being very easy.

**Score 0**: Completely unintelligible to English speakers.
Example: "Je veux manger du pain."
	
**Score 1**: Contains some cognate words, but contains words unintelligible to an English speaker. The cognates might allow them to guess the general topic but not the main idea or actual meaning. Example: "Le maître savant utilise beaucoup de livres." (Has cognates like "savant" but key verbs/objects aren\'t cognates)

**Score 2:** Contains many cognate words. An English speaker might guess the main idea but would miss important details or nuances that change the meaning. Example: "Le patient refuse absolument de prendre ses médicaments malgré les protestations constantes du docteur." [^fn-1]

**Score 3:** Fully understandable through cognates. Use almost exclusively cognate words except for basic connectors. Example: "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain."

(Side Note: I found that using o1-preview is necessary for good-quality scoring. 
Other models -- 4o, 4o-mini, and o1-mini -- had a hard time determining what a cognate was. 
Also, they were too lenient, often assigning scores of 3 to sentences that in my opinion an English speaker wouldn't be able to fully understand.
Even o1-preview sometimes uses words an English speaker wouldn't understand.)

## Generation
To save time and money, I pregenerate all sentences on the site using GPT-4o and o1-preview. 
Unsurprisingly, o1 made much higher quality sentences, but 4o did a pretty good job too, and any bad-quality sentences are flagged by the scoring system anyway. So it's OK to cut costs and use a cheaper model when generating sentences, but not when scoring them.
To generate sentences, I made a function `generate_story` that takes in a target difficulty and then asks GPT-4o to generate a story consisting of sentences at that difficulty. This allows me to create a variety of sentences at different difficulty levels to suit the user's needs.

To make the final set of sentences seen on the site, my script repeatedly calls, and saves the output of, `generate_story` with the target difficulty set to a randomly-generated integer between 0 and 3, inclusive. 
Here's a breakdown of how many sentences the site currently has per difficulty level (recall that 1 story = 10 sentences). 

| Score Range | # Sentences Available |
|-------------|-----------------------|
| 0.00-0.99   | 30                    |
| 1.00-1.99   | 110                   |
| 2.00-2.49   | 120                   |
| 2.50-3.00   | 190                   |

For those interested, the exact prompts used to score and generate sentences are below:
```
# Source code: https://github.com/vkethana/cognate_sentences
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
language_codes = {
    'fr': 'French'
}
SENTENCE_GENERATION_MODEL = 'gpt-4o'
SENTENCE_SCORING_MODEL = 'o1-preview' # 'o1' doesn't work for some reason
 
def generate_story(lang_code, num_sentences, target_difficulty):
    system_prompt = f"""
    You are a fluent speaker of both {language_codes[lang_code]} and English.
    Generate exactly {num_sentences} {language_codes[lang_code]} sentences that:
    1. Form a coherent narrative where each sentence follows from the previous one
    2. Target difficulty level {target_difficulty} using these criteria:

        Level 0: Completely unintelligible to English speakers.
        Example: "Je veux manger du pain."

        Level 1: Contains some cognate words, but is largely unintelligible to an English speaker. The cognates might allow them to guess the general topic but not the actual meaning.
        Example: "Le maître savant utilise beaucoup de livres." (Has cognates like "savant" but key verbs/objects aren\'t cognates)

        Level 2: Contains many cognate words. An English speaker could understand the main idea but would miss important details or nuances that change the meaning.
        Example: "Le patient refuse absolument de prendre ses médicaments malgré les protestations constantes du docteur."
        An English speaker would get "patient refuses absolutely to take medications" and "constant protestations doctor" but might miss "his" and "despite", changing their understanding of whose medications and the relationship between the refusal and protestations.

        Level 3: Fully understandable through cognates. Use almost exclusively cognate words except for basic connectors.
        Example: "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain."

        DIFFICULTY TARGETING STRATEGIES:
        Difficulty 0: Use basic, high-frequency native vocabulary, avoid international words
        Difficulty 1: Use 25-30% cognates in non-crucial positions. Has cognates but leaves major meaning gaps.
        Difficulty 2: Use 50-60% cognates in main concept positions. Sentence is mostly understandable but has subtle meaning changes due to missed words\n
        Difficulty 3: Use 80-90% cognates, especially for key meaning-bearing words. Any small connecting words (le, que, etc.) can be ignored without losing meaning. Should be assigned sparingly - only when missed words don\'t change meaning\n

    {% raw %}Format your response as a JSON array of {num_sentences} objects:
    {{
        "sentence": "<Generated sentence>",
        "target_difficulty": {target_difficulty},
        "reasoning": "<Why this sentence matches difficulty. If this is not the first sentence, also explain why this continues the story from the previous sentence in this JSON array.>",
        "cognate_words": [<List of cognates used>]
    }}{% endraw %}

    Important: Each sentence must directly follow from the previous one to form a coherent story.
    Generate {num_sentences} sentences meeting these criteria (difficulty level and story continuation).
    Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
    """
    
    response = client.chat.completions.create(
        model=SENTENCE_GENERATION_MODEL,
        messages=[{'role': 'user', 'content': system_prompt}],
        temperature=1.0
    )
    
    # Parse generated sentences
    return json.loads(response.choices[0].message.content)

def gpt_scored_rubric_batch(sentences):
    '''
    Score multiple French sentences at once using GPT-4.

    Args:
        sentences: List of sentences to score
    Returns:
        List of scoring results
    '''

    system_prompt = f"""
    You are an expert in French to English translation. I will give you {len(sentences)} sentences in French, and I want you to score each of them on a scale from 0-3 using the following rubric:

    0: Completely unintelligible to English speakers.
    Example: "Je veux manger du pain."

    1: Contains some cognate words, but contains words unintelligible to an English speaker. The cognates might allow them to guess the general topic but not the main idea or actual meaning.
    Example: "Le maître savant utilise beaucoup de livres." (Has cognates like "savant" but key verbs/objects aren\'t cognates)

    2: Contains many cognate words. An English speaker might guess the main idea but would miss important details or nuances that change the meaning.
    Example: "Le patient refuse absolument de prendre ses médicaments malgré les protestations constantes du docteur."
    An English speaker would get "patient refuses absolutely to take medications" and "constant protestations doctor" but might miss "his" and "despite", changing their understanding of whose medications and the relationship between the refusal and protestations.

    3: Fully understandable through cognates. Use almost exclusively cognate words except for basic connectors.
    Example: "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain."

    Important scoring notes:
    - Score 0 sentences have little to no cognates
    - Score 1 sentences have cognates but leave major meaning gaps
    - Score 2 sentences are mostly understandable but have subtle meaning changes due to missed words
    - Score 3 should be assigned sparingly - only when missed words don’t change meaning

    {% raw %}For each sentence, provide a JSON object with these fields:
    {{
      "sentence": "<Sentence>",
      "cognate_words": [<List of Cognate Words>],
      "reasoning": "<Reasoning for the score>",
      "score": <Numerical for the Sentence (0-3)>
    }} {% endraw %}

    Please format your response as a JSON array of these objects. You should have {len(sentences)} objects in your array.

    Here are the sentences to score:
    {json.dumps(sentences, ensure_ascii=False)}
    Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
    """

    completion = client.chat.completions.create(
        model=SENTENCE_SCORING_MODEL,
        messages=[
            {'role': 'user', 'content': system_prompt}
        ],
        temperature=1
    )
    
    response_text = completion.choices[0].message.content.strip()
    try:
        results = json.loads(response_text)
        return results
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the response.")
        raise
```
# Approaches that didn't work
- **Sentence starters:** 
I was initially worried that repeatedly asking the model to generate sentences would result in the same stories being generated over and over. 
To deal with this, I modified my prompt to randomly pick a sentence starter from a hardcoded list of unfinished French sentences. I then asked the model to generate sentences which continued off the sentence starter.
This works, but I eventually got rid of it and found that the sentences were still diverse enough.
- **Live generation:**
Rather than pre-generating the sentences, I originally thought about generating them on the spot and feeding the model with information about the user's past performance.
But pre-generating sentences is cheaper, and we can still adapt to the user's performance using the scoring system.
- **Cognate ratios:**
Originally, I scored sentences using a weighted combination of GPT-4's judgments and the percentage of cognate words in the sentence. 
This is a bad idea because it treats all cognate words equally, leading to inaccurate scoring. 
For example, "ouvre" and "technologie" are both cognates, but the latter is much easier to understand.
A possible fix for this problem is to take into account the age of acquisition (AoA) of the cognate words.
I plan to return to this idea, using a system that gives better scores to some cognate words.

# Some optimizations I made
- **Chain of Thought Prompting**: I tell the model to reason through its scoring and generation process. This substantially reduces hallucinations and improves the output quality of weaker models.
For example, my prompt for sentence scoring tells the LM to use the following output in its response:

```json
    {
      "sentence": "<Sentence>",
      "cognate_words": "[<List of Cognate Words>]",
      "reasoning": "<Reasoning for the score>",
      "score": "<Numerical for the Sentence (0-3)>"
    }
```
- **Batching LLM calls to reduce inference costs:** Sentences are generated and scored in batches of 10, which brings down the cost and time of generating and scoring stories a lot.
- **Require JSON outputs:** I wasted a lot of time trying to get the LM to output in a format that was easy to parse in Python. Eventually I realized that JSON outputs were perfect for this situation. 
Anecdotally, it feels like formatting-related hallucinations are less common when the model is tasked with outputting JSON and not an ad-hoc, user-defined format. 

# Findings
With the right prompting and search strategy, LLMs can definitely generate "cognateful" sentences. I think all three of these examples can be mostly understood by an English speaker:

> Netflix est une plateforme de streaming vidéo qui offre une large sélection de films, séries télévisées et documentaires.

> L'intellectuel français Voltaire a dit: "La tolérance est un ingrédient essentiel de la civilisation."
 
> En 1310, le navigateur italien Marco Polo a traversé l'océan Indien.

Some cognate words have a stronger association with high-scoring sentences than others. 
For example, *université* and *enthousiasme* have average scores of 3.00, whereas *recherches* and *ouvre* have average scores of 1.67. 
These findings might seem obvious at first glance, but it's proof that the scoring function is doing something right!
Cognates that are very easy to understand receive high scores. 
More difficult or obscure cognates receive lower scores.
Here's a non-exhaustive table of some cognates and the average scores of the sentences containing them.

| 1.00 - 1.99         | 2.00 - 2.99                                | 3.00              |
|---------------------|--------------------------------------------|-------------------|
| arbre               | internationale                            | université        |
| mystérieux          | succès                                    | applaudissent     |
| Après               | célèbre                                   | admire            |
| impatience          | présente                                  | directeur         |
| forêt               | entier                                    | exposition        |
| ensemble            | Paris                                     | annonce           |
| ouvre               | musée                                     | communauté        |
| contribution        | musicien                                  | invitation        |
| recherches          | moderne                                   | accepte           |
| chat                | nouvelle                                  | enthousiasme      |
| Thomas              | organise                                  | révolutionnaire   |
| cuisine             | principal                                 | invite            |
| porte               | problème                                  | technologie       |
| lit                 | académique                                | immédiatement     |
| Luc                 | économique                                | planifier         |
| soleil              | voyager                                   | collection        |
| mais                | secret                                    | objet             |
| entre               | performance                               | éducation         |
| livre               | formule                                   | thème             |
| cherche             | incroyable                                |                   |
|                     | monde                                     |                   |
|                     | professeur                                |                   |
|                     | conférence                                |                   |

# Features I plan to add
- Scale up the number of sentences in the app.
- Bring back beam search for sentence generation: Currently I'm making stories by generating 10 sentences at once. A better, but slower and more costly, way to get high-scoring sentences is to generate many options, expand the highest-scoring ones, and discard the rest, gradually building up the stories.
- Remove all English from the UI. Instead, express UI functions using images and icons. Any words which appear on the screen should be in the target language, not English, in order to immerse the user as much as possible.
- Come up with better heuristics for bumping up and down the user's difficulty score based on their performance. Right now, we simply decrement / increment the user's difficulty by 0.10 for each correct or incorrect answer. (Note that lower difficulty values = harder, not easier, sentences)
- **Improve sentence scoring:** I think that this is the hardest part of the project and that the sentence scoring has a lot of room for improvement. 
For example, I could modify the scoring system to use a weighted combination[^fn-2] of two things: GPT-4 judgement scoring and the presence of certain high-scoring cognate words (see "Findings" above).
- Add support for languages other than English.

# How you can help
If you're familiar with NLP, linguistics, or software development, you can help out by suggesting solutions to the following blockers that I'm currently facing.
- **Cheaper and faster scoring**:
Is there a cheaper, more scalable way to score sentences than what I've described here?
To recap, using reasoning models like o1 is effective but slow and expensive.
Using models other than o1 results in bad quality sentences.
Using non LLM-powered scoring misses the nuances of what makes a sentence easy or hard to understand. 
- **More intuitive UI**: Users should be able to understand how the app works without having to read several hundred words about it. How can we engineer the UI so that it's immediately obvious how to use the app and why it will teach you French?
- **Better gameplay loop**: Right now, all the user does is read sentences, translate them, and watch their score go up or down.
How can we make the app more fun?

----- 
[^fn-1]: Justification: An English speaker would get "patient refuses absolutely to take medications" and "constant protestations doctor" but might miss "his" and "despite", changing their understanding of whose medications and the relationship between the refusal and protestations.
[^fn-2]: Special thanks to PhD student Nicholas Tomlin for suggesting this system for sentence scoring, as well as many other helpful ideas regarding the UI and sentence generation.
