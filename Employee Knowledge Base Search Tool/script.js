document.getElementById('searchButton').addEventListener('click', async () => {
    const question = document.getElementById('questionInput').value;
    if (question) {
        document.getElementById('results').innerHTML = 'Loading...';
        try {
            const index = await getIndex();
            const results = getSimilaritySearchResults(index, question);
            displayResults(results);
            const embedding = getEmbedding(question);
            displayEmbedding(embedding);
        } catch (error) {
            document.getElementById('results').innerHTML = `Error: ${error.message}`;
        }
    }
});

async function getIndex() {
    // Simulate index creation
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve('index');
        }, 1000);
    });
}

function getSimilaritySearchResults(index, question) {
    // Simulate similarity search
    return [
        { content: 'Sample result 1', score: 0.9 },
        { content: 'Sample result 2', score: 0.8 }
    ];
}

function getEmbedding(text) {
    // Simulate embedding generation
    return { embedding: [0.1, 0.2, 0.3] };
}

function displayResults(results) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>Search Results</h2>';
    results.forEach(result => {
        const resultDiv = document.createElement('div');
        resultDiv.innerHTML = `<p>${result.content} (Score: ${result.score})</p>`;
        resultsDiv.appendChild(resultDiv);
    });
}

function displayEmbedding(embedding) {
    const embeddingDiv = document.getElementById('embedding');
    embeddingDiv.classList.remove('hidden');
    document.getElementById('embeddingJson').textContent = JSON.stringify(embedding, null, 2);
}
