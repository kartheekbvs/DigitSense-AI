document.addEventListener('DOMContentLoaded', () => {
    // Canvas Setup
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');

    // UI Elements
    const steps = {
        cnn: document.getElementById('step-cnn'),
        pca: document.getElementById('step-pca'),
        lr: document.getElementById('step-lr')
    };
    const resultContainer = document.getElementById('result-container');
    const predictedDigitEl = document.getElementById('predicted-digit');
    const confidenceList = document.getElementById('confidence-list');

    // Canvas State
    let isDrawing = false;
    let hasDrawing = false;

    // Drawing parameters
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000000';

    function startPosition(e) {
        isDrawing = true;
        hasDrawing = true;
        resetPipelineUI();
        draw(e);
    }

    function endPosition() {
        isDrawing = false;
        ctx.beginPath();
    }

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        const clientY = e.clientY || (e.touches && e.touches[0].clientY);
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }

    canvas.addEventListener('mousedown', startPosition);
    canvas.addEventListener('mouseup', endPosition);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseout', endPosition);
    canvas.addEventListener('touchstart', startPosition, { passive: false });
    canvas.addEventListener('touchend', endPosition);
    canvas.addEventListener('touchmove', draw, { passive: false });

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        hasDrawing = false;
        resetPipelineUI();
    }
    clearBtn.addEventListener('click', clearCanvas);

    async function animatePipelineStep(stepElement, duration) {
        stepElement.classList.remove('completed');
        stepElement.classList.add('active');
        const progressBar = stepElement.querySelector('.progress-bar');
        progressBar.style.width = '0%';

        return new Promise(resolve => {
            let start = null;
            function step(timestamp) {
                if (!start) start = timestamp;
                const progress = timestamp - start;
                const percentage = Math.min((progress / duration) * 100, 100);
                progressBar.style.width = `${percentage}%`;
                if (progress < duration) {
                    window.requestAnimationFrame(step);
                } else {
                    stepElement.classList.remove('active');
                    stepElement.classList.add('completed');
                    resolve();
                }
            }
            window.requestAnimationFrame(step);
        });
    }

    function renderResults(predictionData) {
        predictedDigitEl.textContent = predictionData.prediction;
        confidenceList.innerHTML = '';

        // Take top 5 confidences
        const top5 = predictionData.confidences.slice(0, 5);

        top5.forEach((item, index) => {
            const isTop = index === 0;
            const el = document.createElement('div');
            el.className = `confidence-item ${isTop ? 'top-prediction' : ''}`;
            el.innerHTML = `
                <span class="conf-label">${item.digit}</span>
                <div class="conf-bar-wrapper">
                    <div class="conf-bar" style="width: 0%"></div>
                </div>
                <span class="conf-value">${item.conf}%</span>
            `;
            confidenceList.appendChild(el);
            setTimeout(() => {
                el.querySelector('.conf-bar').style.width = `${item.conf}%`;
            }, 50 + (index * 100));
        });

        resultContainer.classList.add('show');
    }

    function resetPipelineUI() {
        Object.values(steps).forEach(step => {
            step.classList.remove('active', 'completed');
            const pb = step.querySelector('.progress-bar');
            if (pb) pb.style.width = '0%';
        });
        resultContainer.classList.remove('show');
        predictedDigitEl.textContent = '-';
        confidenceList.innerHTML = '';
    }

    predictBtn.addEventListener('click', async () => {
        if (!hasDrawing) {
            alert("Please draw a digit first!");
            return;
        }

        predictBtn.disabled = true;
        clearBtn.disabled = true;
        resetPipelineUI();

        try {
            // Parallel fetch to reduce perceived latency
            const imageData = canvas.toDataURL('image/png');
            const apiCall = fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            }).then(r => r.json());

            // Visual Pipeline Sequence
            await animatePipelineStep(steps.cnn, 800);
            await animatePipelineStep(steps.pca, 600);
            await animatePipelineStep(steps.lr, 400);

            const result = await apiCall;
            if (result.error) {
                alert(`Error: ${result.error}`);
            } else {
                renderResults(result);
            }
        } catch (error) {
            console.error(error);
            alert("Connection error. Is the Flask server running?");
        } finally {
            predictBtn.disabled = false;
            clearBtn.disabled = false;
        }
    });
});
