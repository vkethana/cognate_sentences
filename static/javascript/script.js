// Constants
const DIFFICULTY_STEP_DOWN = 0.1;
const DIFFICULTY_STEP_UP = 0.1;
const MAX_DIFFICULTY = 3.0;
const MIN_DIFFICULTY = 0.0;

// State variables
let currentSentenceToTranslate = '';

/*
// Initialize local storage and UI state
if (!localStorage.getItem('currentStoryFile')) {
    localStorage.setItem('currentStoryFile', '');
    localStorage.setItem('currentSentenceIndex', '0');
    localStorage.setItem('userDifficulty', '2.5');
    localStorage.setItem('seenStories', JSON.stringify([]));
}
*/
localStorage.setItem('currentStoryFile', '');
localStorage.setItem('currentSentenceIndex', '0');
localStorage.setItem('userDifficulty', '3');
localStorage.setItem('seenStories', JSON.stringify([]));

// Set initial stats panel display state explicitly
document.getElementById('stats-panel').style.display = 'none';

// Stats display initialization
document.getElementById('user-difficulty').textContent = 
    parseFloat(localStorage.getItem('userDifficulty')).toFixed(2);
document.getElementById('stories-seen').textContent = 
    JSON.parse(localStorage.getItem('seenStories')).length;

// UI Functions
function toggleStats() {
    const panel = document.getElementById('stats-panel');
    const isVisible = panel.style.display !== 'none';
    panel.style.display = isVisible ? 'none' : 'block';

    // Toggle difficulty displays
    /*
    const difficulties = document.querySelectorAll('.difficulty');
    
    difficulties.forEach(diff => {
        diff.classList.toggle('visible');
    });
    */
    document.body.classList.toggle('show-stats');
}

function createSentenceBlock(sentence, difficulty, isActive = false) {
    const block = document.createElement('div');
    block.className = `sentence-block ${isActive ? 'active' : ''}`;
    
    const frenchContainer = document.createElement('div');
    frenchContainer.className = 'french-container';
    frenchContainer.innerHTML = `
        <span class="flag">🇫🇷</span>
        <span class="sentence">${sentence}</span>
        <span class="difficulty">📊 ${parseFloat(difficulty).toFixed(2)}</span>
    `;
    block.appendChild(frenchContainer);

    if (isActive) {
        const translationContainer = document.createElement('div');
        translationContainer.className = 'translation-container';
        translationContainer.innerHTML = `
            <span class="flag">🇬🇧</span>
            <input type="text" class="translation-input" placeholder="...">
            <button onclick="checkTranslation(this)" title="Check translation">✓</button>
            <!-- DEBUG BUTTON - Remove in production -->
            <button onclick="forceSkip(this)" title="Force skip" class="debug-button">⏭️</button>
        `;
        block.appendChild(translationContainer);
    }

    return block;
}

// Core Logic Functions
function adjustDifficulty(correct) {
    let currentDifficulty = parseFloat(localStorage.getItem('userDifficulty'));
    if (correct) {
        currentDifficulty = Math.max(MIN_DIFFICULTY, currentDifficulty - DIFFICULTY_STEP_DOWN);
    } else {
        currentDifficulty = Math.min(MAX_DIFFICULTY, currentDifficulty + DIFFICULTY_STEP_UP);
    }
    localStorage.setItem('userDifficulty', currentDifficulty.toString());
    document.getElementById('user-difficulty').textContent = currentDifficulty.toFixed(2);
}

async function handleNext() {
    const activeBlock = document.querySelector('.sentence-block.active');
    if (!activeBlock) {
        await handleSpacePress();
    }
}

document.addEventListener('DOMContentLoaded', function() {
    handleSpacePress();
});

async function handleSpacePress() {
    const display = document.getElementById('sentence-display');
    
    // Clear start prompt if it exists
    const startPrompt = display.querySelector('.start-prompt');
    if (startPrompt) {
        display.innerHTML = '';
    }

    const currentStoryFile = localStorage.getItem('currentStoryFile');
    let currentSentenceIndex = parseInt(localStorage.getItem('currentSentenceIndex'));

    const response = await fetch('/get-sentence', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(
            currentStoryFile === '' ?
            { 
                needNewStory: true,
                userDifficulty: parseFloat(localStorage.getItem('userDifficulty')),
                seenStories: JSON.parse(localStorage.getItem('seenStories'))
            } :
            { 
                storyFile: currentStoryFile,
                sentenceIndex: currentSentenceIndex
            }
        )
    });

    const data = await response.json();
    
    if (data.isLastSentence) {
        const seenStories = JSON.parse(localStorage.getItem('seenStories'));
        seenStories.push(currentStoryFile);
        localStorage.setItem('seenStories', JSON.stringify(seenStories));
        document.getElementById('stories-seen').textContent = seenStories.length;
        
        // Add story end marker without clearing existing sentences
        const endMarker = document.createElement('div');
        endMarker.className = 'story-end';
        endMarker.innerHTML = `
            <div>🏁</div>
            <button onclick="startNewStory()" class="next-button" title="Start new story">➡️</button>
        `;
        display.appendChild(endMarker);
        return;
    }

    // Create and append new sentence block
    const sentenceBlock = createSentenceBlock(data.sentence, data.sentenceDifficulty, true);
    display.appendChild(sentenceBlock);
    currentSentenceToTranslate = data.sentence;

    // Update storage
    if (data.storyFile) {
        localStorage.setItem('currentStoryFile', data.storyFile);
        localStorage.setItem('currentSentenceIndex', '1');
        document.getElementById('story-difficulty').textContent = data.storyDifficulty.toFixed(2);
    } else {
        localStorage.setItem('currentSentenceIndex', (currentSentenceIndex + 1).toString());
    }

    // Scroll to bottom
    window.scrollTo(0, document.body.scrollHeight);
}

async function startNewStory() {
    localStorage.setItem('currentStoryFile', '');
    localStorage.setItem('currentSentenceIndex', '0');
    document.getElementById('story-difficulty').textContent = '-';
    document.getElementById('sentence-display').innerHTML = '';
    await handleSpacePress();
}

async function checkTranslation(button) {
    const block = button.closest('.sentence-block');
    const input = block.querySelector('.translation-input');
    const translation = input.value.trim();
    
    if (!translation) {
        block.classList.add('incorrect');
        setTimeout(() => {
            block.classList.remove('incorrect');
            block.classList.add('active');
        }, 1000);
        return;
    }

    try {
        const response = await fetch('/score_translation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                original: currentSentenceToTranslate,
                translation: translation
            })
        });

        const data = await response.json();
        
        if (data.isCorrect) {
            adjustDifficulty(true);
            block.classList.remove('active', 'incorrect');
            block.classList.add('correct');
            
            const completedTranslation = document.createElement('div');
            completedTranslation.className = 'completed-translation';
            completedTranslation.textContent = translation;
            block.appendChild(completedTranslation);

            block.querySelector('.translation-container').remove();
            handleSpacePress();
        } else {
            adjustDifficulty(false);
            block.classList.remove('correct');
            block.classList.add('incorrect');
            
            if (data.wrongMorphemes && data.wrongMorphemes.length > 0) {
                const frenchSentence = block.querySelector('.sentence');
                let sentenceHtml = currentSentenceToTranslate;
                data.wrongMorphemes.forEach(morpheme => {
                    sentenceHtml = sentenceHtml.replace(
                        morpheme,
                        `<strong style="color: #dc3545">${morpheme}</strong>`
                    );
                });
                frenchSentence.innerHTML = sentenceHtml;
            }
            
            setTimeout(() => {
                block.classList.remove('incorrect');
                block.classList.add('active');
            }, 1000);
        }
    } catch (error) {
        console.error('Error checking translation:', error);
        block.classList.add('incorrect');
        setTimeout(() => {
            block.classList.remove('incorrect');
            block.classList.add('active');
        }, 1000);
    }
}

async function forceSkip(button) {
    const block = button.closest('.sentence-block');
    block.querySelector('.translation-container').remove();
    block.classList.remove('active');
    await handleSpacePress();
}

// Event Listeners
document.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && e.target.classList.contains('translation-input')) {
        const button = e.target.nextElementSibling;
        if (button) {
            button.click();
        }
    }
});
