/**
 * Eigen-stuff Calculator
 * Calculates eigenvalues and eigenspace bases for n×n matrices (n ≤ 5).
 */

// ==========================================================================
// DOM Elements
// ==========================================================================
const dimensionSlider = document.getElementById('dimension-slider');
const dimensionDisplay = document.getElementById('dimension-display');
const matrixGrid = document.getElementById('matrix-grid');
const computeBtn = document.getElementById('compute-btn');
const resultsSection = document.getElementById('results-section');
const errorDisplay = document.getElementById('error-display');

// ==========================================================================
// State
// ==========================================================================
let currentN = parseInt(dimensionSlider.value, 10);

// ==========================================================================
// Matrix Grid Generation
// ==========================================================================

/**
 * Generates the matrix input grid based on the current dimension.
 * @param {number} n - The dimension of the matrix.
 */
function generateGrid(n) {
    matrixGrid.innerHTML = '';
    matrixGrid.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
    
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'matrix-input';
            input.id = `cell-${i}-${j}`;
            input.placeholder = '0';
            input.setAttribute('aria-label', `Row ${i + 1}, Column ${j + 1}`);
            input.step = 'any'; // Allow decimals
            matrixGrid.appendChild(input);
        }
    }
}

/**
 * Reads the matrix values from the DOM.
 * @returns {number[][]} - A 2D array representing the matrix.
 */
function getMatrix() {
    const matrix = [];
    for (let i = 0; i < currentN; i++) {
        const row = [];
        for (let j = 0; j < currentN; j++) {
            const input = document.getElementById(`cell-${i}-${j}`);
            const value = parseFloat(input.value);
            row.push(isNaN(value) ? 0 : value);
        }
        matrix.push(row);
    }
    return matrix;
}

// ==========================================================================
// Computation Logic
// ==========================================================================

/**
 * Formats a complex number for display.
 * @param {math.Complex | number} num - The number to format.
 * @returns {string} - A human-readable string (e.g., "3 + 2i").
 */
function formatComplex(num) {
    if (typeof num === 'number') {
        return formatReal(num);
    }
    
    const re = num.re;
    const im = num.im;
    
    // Tolerance for treating near-zero as zero
    const tol = 1e-10;
    const reZero = Math.abs(re) < tol;
    const imZero = Math.abs(im) < tol;
    
    if (reZero && imZero) return '0';
    if (imZero) return formatReal(re);
    if (reZero) {
        if (Math.abs(im - 1) < tol) return 'i';
        if (Math.abs(im + 1) < tol) return '−i';
        return `${formatReal(im)}i`;
    }
    
    // Both parts non-zero
    const imSign = im >= 0 ? '+' : '−';
    const absIm = Math.abs(im);
    const imStr = Math.abs(absIm - 1) < tol ? '' : formatReal(absIm);
    
    return `${formatReal(re)} ${imSign} ${imStr}i`;
}

/**
 * Formats a real number, rounding to sensible precision.
 * @param {number} num - The number to format.
 * @returns {string}
 */
function formatReal(num) {
    // Round to 6 decimal places to avoid floating point noise
    const rounded = Math.round(num * 1e6) / 1e6;
    
    // If it's effectively an integer, show it as one
    if (Number.isInteger(rounded)) {
        return rounded.toString();
    }
    
    return rounded.toString();
}

/**
 * Checks if two complex numbers are approximately equal.
 */
function complexApproxEqual(a, b, tol = 1e-9) {
    const aRe = typeof a === 'number' ? a : a.re;
    const aIm = typeof a === 'number' ? 0 : a.im;
    const bRe = typeof b === 'number' ? b : b.re;
    const bIm = typeof b === 'number' ? 0 : b.im;
    
    return Math.abs(aRe - bRe) < tol && Math.abs(aIm - bIm) < tol;
}

/**
 * Computes eigenvalues and eigenvectors, then groups by eigenvalue.
 * 
 * math.eigs() returns:
 * - values: array of eigenvalues
 * - vectors: matrix where COLUMNS are eigenvectors (column i corresponds to values[i])
 * 
 * @param {number[][]} matrix 
 * @returns {{ eigenvalue: any, eigenvectors: any[], algebraicMultiplicity: number }[]}
 */
function computeEigenData(matrix) {
    const result = math.eigs(matrix);
    
    // Get values as a plain array
    const valuesArray = result.values.toArray ? result.values.toArray() : 
                        (Array.isArray(result.values) ? result.values : result.values.valueOf());
    
    // Get vectors matrix and extract columns
    const vectorsMatrix = result.vectors.toArray ? result.vectors.toArray() : 
                          (Array.isArray(result.vectors) ? result.vectors : result.vectors.valueOf());
    
    const n = valuesArray.length;
    
    // Extract column vectors from the matrix
    // vectorsMatrix[row][col] -> we want column vectors
    const eigenvectorsList = [];
    for (let col = 0; col < n; col++) {
        const vec = [];
        for (let row = 0; row < n; row++) {
            vec.push(vectorsMatrix[row][col]);
        }
        eigenvectorsList.push(vec);
    }
    
    // Group eigenvectors by their eigenvalue
    const eigenMap = new Map();
    
    for (let i = 0; i < n; i++) {
        const eigenvalue = valuesArray[i];
        const eigenvector = eigenvectorsList[i];
        
        // Find existing group or create new one
        let found = false;
        for (const [key, data] of eigenMap) {
            if (complexApproxEqual(key, eigenvalue)) {
                data.eigenvectors.push(eigenvector);
                data.algebraicMultiplicity++;
                found = true;
                break;
            }
        }
        
        if (!found) {
            eigenMap.set(eigenvalue, {
                eigenvalue: eigenvalue,
                eigenvectors: [eigenvector],
                algebraicMultiplicity: 1
            });
        }
    }
    
    return Array.from(eigenMap.values());
}

// ==========================================================================
// Result Rendering
// ==========================================================================

/**
 * Renders a single eigenvector as a column vector UI element.
 * @param {Array} vector - The eigenvector components.
 * @returns {HTMLElement}
 */
function renderEigenvector(vector) {
    const container = document.createElement('div');
    container.className = 'eigenvector';
    
    const leftBracket = document.createElement('span');
    leftBracket.className = 'eigenvector__bracket';
    leftBracket.textContent = '[';
    
    const components = document.createElement('div');
    components.className = 'eigenvector__components';
    
    const vecArray = vector.toArray ? vector.toArray() : vector;
    vecArray.forEach(comp => {
        const compEl = document.createElement('div');
        compEl.className = 'eigenvector__component';
        compEl.textContent = formatComplex(comp);
        components.appendChild(compEl);
    });
    
    const rightBracket = document.createElement('span');
    rightBracket.className = 'eigenvector__bracket';
    rightBracket.textContent = ']';
    
    container.appendChild(leftBracket);
    container.appendChild(components);
    container.appendChild(rightBracket);
    
    return container;
}

/**
 * Renders a result card for a single eigenvalue and its eigenspace.
 * @param {{ eigenvalue: any, eigenvectors: any[], algebraicMultiplicity: number }} data 
 * @param {number} index - For staggered animation delay.
 * @returns {HTMLElement}
 */
function renderResultCard(data, index) {
    const card = document.createElement('article');
    card.className = 'result-card';
    card.style.animationDelay = `${index * 100}ms`;
    
    // Header with eigenvalue
    const header = document.createElement('div');
    header.className = 'result-card__header';
    
    const label = document.createElement('span');
    label.className = 'result-card__label';
    label.textContent = 'Eigenvalue λ =';
    
    const value = document.createElement('span');
    value.className = 'result-card__value';
    value.textContent = formatComplex(data.eigenvalue);
    
    const multiplicity = document.createElement('span');
    multiplicity.className = 'result-card__multiplicity';
    const geoMult = data.eigenvectors.length;
    const algMult = data.algebraicMultiplicity;
    multiplicity.textContent = `(Algebraic: ${algMult}, Geometric: ${geoMult})`;
    
    header.appendChild(label);
    header.appendChild(value);
    header.appendChild(multiplicity);
    
    // Eigenspace section
    const eigenspaceLabel = document.createElement('div');
    eigenspaceLabel.className = 'eigenspace-label';
    eigenspaceLabel.textContent = `Basis for Eigenspace E(${formatComplex(data.eigenvalue)}):`;
    
    const eigenspaceVectors = document.createElement('div');
    eigenspaceVectors.className = 'eigenspace-vectors';
    
    data.eigenvectors.forEach(vec => {
        eigenspaceVectors.appendChild(renderEigenvector(vec));
    });
    
    card.appendChild(header);
    card.appendChild(eigenspaceLabel);
    card.appendChild(eigenspaceVectors);
    
    return card;
}

/**
 * Renders all results to the results section.
 * @param {{ eigenvalue: any, eigenvectors: any[], algebraicMultiplicity: number }[]} eigenData 
 */
function renderResults(eigenData) {
    resultsSection.innerHTML = '';
    
    if (eigenData.length === 0) {
        showError('No eigenvalues found. The matrix may be singular or ill-conditioned.');
        return;
    }
    
    eigenData.forEach((data, index) => {
        resultsSection.appendChild(renderResultCard(data, index));
    });
}

// ==========================================================================
// Error Handling
// ==========================================================================

function showError(message) {
    errorDisplay.textContent = message;
    errorDisplay.setAttribute('aria-hidden', 'false');
}

function hideError() {
    errorDisplay.textContent = '';
    errorDisplay.setAttribute('aria-hidden', 'true');
}

// ==========================================================================
// Event Handlers
// ==========================================================================

function handleDimensionChange() {
    currentN = parseInt(dimensionSlider.value, 10);
    dimensionDisplay.textContent = currentN;
    generateGrid(currentN);
    resultsSection.innerHTML = '';
    hideError();
}

function handleCompute() {
    hideError();
    resultsSection.innerHTML = '';
    
    try {
        const matrix = getMatrix();
        const eigenData = computeEigenData(matrix);
        renderResults(eigenData);
    } catch (err) {
        console.error('Computation error:', err);
        showError(`Computation failed: ${err.message || 'Unknown error'}`);
    }
}

// ==========================================================================
// Initialization
// ==========================================================================

function init() {
    // Set up event listeners
    dimensionSlider.addEventListener('input', handleDimensionChange);
    computeBtn.addEventListener('click', handleCompute);
    
    // Allow Enter key to trigger computation when in an input
    matrixGrid.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            handleCompute();
        }
    });
    
    // Initial grid generation
    generateGrid(currentN);
}

// Run on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
