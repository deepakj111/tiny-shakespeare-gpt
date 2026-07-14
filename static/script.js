document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const temperatureInput = document.getElementById('temperature');
    const tempVal = document.getElementById('temp-val');
    const maxTokensInput = document.getElementById('max-tokens');
    const tokensVal = document.getElementById('tokens-val');
    const topKInput = document.getElementById('top-k');
    const topKVal = document.getElementById('top-k-val');
    const seedInput = document.getElementById('seed');
    
    const outputBox = document.getElementById('output-box');
    const generateForm = document.getElementById('generate-form');
    const promptInput = document.getElementById('prompt-input');
    const generateBtn = document.getElementById('generate-btn');
    const btnText = generateBtn.querySelector('span');
    const btnLoader = document.getElementById('btn-loader');

    // Update value displays
    temperatureInput.addEventListener('input', (e) => tempVal.textContent = e.target.value);
    maxTokensInput.addEventListener('input', (e) => tokensVal.textContent = e.target.value);
    topKInput.addEventListener('input', (e) => topKVal.textContent = e.target.value);

    // Auto-resize textarea
    promptInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') {
            this.style.height = 'auto'; // Reset
        }
    });

    // Enter to submit (Shift+Enter for newline)
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!generateBtn.disabled) {
                generateForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    generateForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const promptText = promptInput.value;
        const temperature = parseFloat(temperatureInput.value);
        const maxTokens = parseInt(maxTokensInput.value, 10);
        const topK = parseInt(topKInput.value, 10);
        const seed = parseInt(seedInput.value, 10);

        // Update UI state
        outputBox.innerHTML = '';
        generateBtn.disabled = true;
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');

        // Add prompt to output box if it's not empty
        if (promptText.trim() !== '') {
            outputBox.textContent = promptText;
        }

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: promptText,
                    max_new_tokens: maxTokens,
                    temperature: temperature,
                    top_k: topK,
                    stream: true,
                    seed: seed
                })
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${await response.text()}`);
            }

            // Handle SSE stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                
                // Keep the last partial line in the buffer
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        if (dataStr === '[DONE]') {
                            break;
                        }
                        try {
                            const data = JSON.parse(dataStr);
                            if (data.text) {
                                outputBox.textContent += data.text;
                                // Auto scroll to bottom
                                outputBox.scrollTop = outputBox.scrollHeight;
                            }
                        } catch (err) {
                            console.error('Error parsing SSE data:', err, dataStr);
                        }
                    }
                }
            }

        } catch (error) {
            console.error('Error during generation:', error);
            const errorEl = document.createElement('div');
            errorEl.style.color = '#ef4444';
            errorEl.style.marginTop = '1rem';
            errorEl.textContent = `Error: ${error.message}`;
            outputBox.appendChild(errorEl);
        } finally {
            // Restore UI state
            generateBtn.disabled = false;
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
        }
    });
});
