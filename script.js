/**
 * Eigen-stuff Calculator
 * Calculates eigenvalues and eigenspace bases for n×n matrices (n ≤ 5).
 */

// ==========================================================================
// DOM Elements
// ==========================================================================
const dimensionSelector = document.getElementById('dimension-selector');
const dimensionBtns = document.querySelectorAll('.dimension-btn');
const formatSelector = document.getElementById('format-selector');
const formatBtns = document.querySelectorAll('.format-btn');
const matrixGrid = document.getElementById('matrix-grid');
const computeBtn = document.getElementById('compute-btn');
const clearBtn = document.getElementById('clear-btn');
const resultsSection = document.getElementById('results-section');
const resultsToolbar = document.getElementById('results-toolbar');
const errorDisplay = document.getElementById('error-display');

// ==========================================================================
// State
// ==========================================================================
let currentN = 3; // Default to 3x3
let vectorFormat = 'fractions'; // 'simplified' | 'fractions' | 'normalized'
let lastEigenData = null; // Store last computation for re-rendering
let lastMatrix = null;

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
// Constants
// ==========================================================================
const TOLERANCE = 1e-9;

// ==========================================================================
// Computation Logic
// ==========================================================================

/**
 * Computes the L2 norm (magnitude) of a vector.
 * @param {Array} vec - The vector (array of numbers or complex).
 * @returns {number}
 */
function vectorNorm(vec) {
    let sum = 0;
    for (const comp of vec) {
        const re = typeof comp === 'number' ? comp : comp.re;
        const im = typeof comp === 'number' ? 0 : comp.im;
        sum += re * re + im * im;
    }
    return Math.sqrt(sum);
}

/**
 * Normalizes a vector to unit length.
 * @param {Array} vec
 * @returns {Array}
 */
function normalizeVector(vec) {
    const norm = vectorNorm(vec);
    if (norm < TOLERANCE) return vec; // Avoid division by zero
    return vec.map(comp => {
        if (typeof comp === 'number') {
            return comp / norm;
        }
        return math.complex(comp.re / norm, comp.im / norm);
    });
}

/**
 * Simplifies a vector to show cleaner integer-like values.
 * Finds a scaling factor that makes components closer to integers.
 * Enforces canonical sign: first non-zero component is positive.
 * For complex vectors, returns the original vector without simplification.
 * @param {Array} vec - The vector to simplify.
 * @returns {Array} - Simplified vector.
 */
function simplifyVector(vec) {
    // Check if vector has significant complex components
    const hasComplexParts = vec.some(v => 
        v && typeof v === 'object' && 'im' in v && Math.abs(v.im) > 1e-6
    );
    
    // Don't simplify complex vectors - integer ratio scaling doesn't apply
    if (hasComplexParts) {
        return vec;
    }
    
    // Get real values from vector
    const vals = vec.map(v => typeof v === 'number' ? v : v.re);
    
    // Filter out near-zero values
    const nonZero = vals.filter(v => Math.abs(v) > 1e-6);
    if (nonZero.length === 0) return vec;
    
    // Find the smallest non-zero absolute value
    const minAbs = Math.min(...nonZero.map(Math.abs));
    
    // Scale so smallest non-zero becomes 1
    let scaled = vals.map(v => Math.abs(v) < 1e-6 ? 0 : v / minAbs);
    
    // Check if all scaled values are close to integers
    const allIntegers = scaled.every(v => Math.abs(v - Math.round(v)) < 0.01);
    
    let result;
    if (allIntegers) {
        // Return integer-scaled values
        result = scaled.map(v => Math.round(v));
    } else {
        // Try scaling by common multipliers to find integer ratios
        let found = false;
        for (const mult of [2, 3, 4, 5, 6]) {
            const testScaled = vals.map(v => Math.abs(v) < 1e-6 ? 0 : (v / minAbs) * mult);
            const allInt = testScaled.every(v => Math.abs(v - Math.round(v)) < 0.05);
            if (allInt) {
                result = testScaled.map(v => Math.round(v));
                found = true;
                break;
            }
        }
        
        if (!found) {
            // Fall back to the min-scaled version (might have decimals)
            result = scaled.map(v => {
                if (Math.abs(v) < 1e-6) return 0;
                if (Math.abs(v - Math.round(v)) < 0.01) return Math.round(v);
                return v;
            });
        }
    }
    
    // Enforce canonical sign: first non-zero component should be positive
    return makeCanonicalSign(result);
}

/**
 * Enforces canonical sign convention: first non-zero component is positive.
 * @param {Array} vec - The vector to normalize sign.
 * @returns {Array} - Vector with canonical sign.
 */
function makeCanonicalSign(vec) {
    // Find first non-zero component
    for (let i = 0; i < vec.length; i++) {
        const val = typeof vec[i] === 'number' ? vec[i] : (vec[i].re || 0);
        if (Math.abs(val) > 1e-9) {
            if (val < 0) {
                // Flip signs of all components
                return vec.map(v => {
                    if (typeof v === 'number') return -v;
                    if (v && typeof v === 'object' && 're' in v) {
                        return math.complex(-v.re, -v.im);
                    }
                    return v;
                });
            }
            return vec; // Already canonical
        }
    }
    return vec; // All zeros
}

/**
 * Converts a simplified integer vector to fraction form.
 * Divides all components by the last non-zero value (so last term = 1).
 * Returns array of {numerator, denominator} or formatted strings.
 * @param {Array} vec - The simplified integer vector.
 * @returns {Array} - Vector with fractional components as strings.
 */
function vectorToFractions(vec) {
    // Check if vector has significant complex components
    const hasComplexParts = vec.some(v => 
        v && typeof v === 'object' && 'im' in v && Math.abs(v.im) > 1e-6
    );
    
    if (hasComplexParts) {
        // For complex vectors, just return formatted complex numbers
        return vec.map(v => formatComplex(v));
    }
    
    // Get real values
    const vals = vec.map(v => typeof v === 'number' ? v : v.re);
    
    // Find the last non-zero value
    let lastNonZeroIdx = -1;
    for (let i = vals.length - 1; i >= 0; i--) {
        if (Math.abs(vals[i]) > 1e-9) {
            lastNonZeroIdx = i;
            break;
        }
    }
    
    if (lastNonZeroIdx === -1) {
        // All zeros
        return vals.map(() => '0');
    }
    
    const divisor = vals[lastNonZeroIdx];
    
    // Divide all by the last non-zero value
    return vals.map(v => {
        if (Math.abs(v) < 1e-9) return '0';
        
        const ratio = v / divisor;
        
        // Check if it's a clean integer
        if (Math.abs(ratio - Math.round(ratio)) < 1e-9) {
            return Math.round(ratio).toString();
        }
        
        // Use math.js to get a nice fraction representation
        try {
            const frac = math.fraction(ratio);
            if (frac.d === 1) {
                return frac.n.toString();
            }
            // Handle negative fractions properly
            const sign = frac.s < 0 ? '-' : '';
            return `${sign}${Math.abs(frac.n)}/${frac.d}`;
        } catch (e) {
            // Fallback to decimal
            return ratio.toFixed(4);
        }
    });
}

/**
 * Computes the verification residual: || A*v - λ*v ||
 * Uses math.js for proper complex arithmetic support.
 * @param {number[][]} matrix - The original matrix A.
 * @param {Array} eigenvector - The eigenvector v.
 * @param {number | math.Complex} eigenvalue - The eigenvalue λ.
 * @returns {number} - The residual norm.
 */
function computeResidual(matrix, eigenvector, eigenvalue) {
    const n = matrix.length;
    
    // Ensure eigenvector is a flat 1D array
    let vec = eigenvector;
    if (vec.toArray) vec = vec.toArray();
    if (Array.isArray(vec) && vec.length > 0 && Array.isArray(vec[0])) {
        // It's a 2D matrix (column vector), flatten it
        vec = vec.map(row => Array.isArray(row) ? row[0] : row);
    }
    
    // Check if we're dealing with complex values
    const isComplexEigenvalue = eigenvalue && typeof eigenvalue === 'object' && 'im' in eigenvalue && Math.abs(eigenvalue.im) > TOLERANCE;
    const hasComplexVector = vec.some(v => v && typeof v === 'object' && 'im' in v && Math.abs(v.im) > TOLERANCE);
    
    if (isComplexEigenvalue || hasComplexVector) {
        // Use math.js for complex arithmetic
        try {
            // Convert matrix, vector, and eigenvalue to math.js format
            const A = math.matrix(matrix);
            const v = math.matrix(vec.map(c => typeof c === 'number' ? math.complex(c, 0) : c));
            const lambda = typeof eigenvalue === 'number' ? math.complex(eigenvalue, 0) : eigenvalue;
            
            // Compute A*v - λ*v
            const Av = math.multiply(A, v);
            const lambdaV = math.multiply(lambda, v);
            const diff = math.subtract(Av, lambdaV);
            
            // Compute the norm of the difference
            const diffArray = diff.toArray ? diff.toArray() : diff;
            let sumSq = 0;
            for (const comp of diffArray) {
                const re = typeof comp === 'number' ? comp : (comp.re || 0);
                const im = typeof comp === 'number' ? 0 : (comp.im || 0);
                sumSq += re * re + im * im;
            }
            return Math.sqrt(sumSq);
        } catch (e) {
            console.warn('Complex residual computation failed:', e.message);
            // Fall back to real-only computation
        }
    }
    
    // Real-only computation (faster for real cases)
    const Av = [];
    for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) {
            const m = typeof matrix[i][j] === 'number' ? matrix[i][j] : matrix[i][j].re;
            const vVal = typeof vec[j] === 'number' ? vec[j] : vec[j].re;
            sum += m * vVal;
        }
        Av.push(sum);
    }
    
    // λ * v
    const lambdaVal = typeof eigenvalue === 'number' ? eigenvalue : eigenvalue.re;
    const lambdaV = vec.map(v => {
        const val = typeof v === 'number' ? v : v.re;
        return lambdaVal * val;
    });
    
    // Compute |Av - λv|
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
        const diff = Av[i] - lambdaV[i];
        sumSq += diff * diff;
    }
    
    return Math.sqrt(sumSq);
}

/**
 * Filters eigenvectors to find a linearly independent basis.
 * Uses a simplified Gram-Schmidt-like approach to detect linear dependency.
 * @param {Array[]} vectors - Array of eigenvectors.
 * @returns {Array[]} - Linearly independent subset.
 */
function filterLinearlyIndependent(vectors) {
    if (vectors.length <= 1) return vectors;
    
    const basis = [];
    
    for (const vec of vectors) {
        // Project vec onto current basis and subtract
        let residual = vec.map(x => x); // Copy
        
        for (const basisVec of basis) {
            // Compute dot product (inner product)
            let dotProduct = 0;
            for (let i = 0; i < vec.length; i++) {
                const vComp = typeof vec[i] === 'number' ? vec[i] : vec[i].re;
                const bComp = typeof basisVec[i] === 'number' ? basisVec[i] : basisVec[i].re;
                dotProduct += vComp * bComp;
            }
            
            // Subtract projection
            residual = residual.map((comp, i) => {
                const bComp = typeof basisVec[i] === 'number' ? basisVec[i] : basisVec[i].re;
                if (typeof comp === 'number') {
                    return comp - dotProduct * bComp;
                }
                return math.complex(comp.re - dotProduct * bComp, comp.im);
            });
        }
        
        // Check if residual is significant (not near-zero)
        const residualNorm = vectorNorm(residual);
        if (residualNorm > TOLERANCE * 100) {
            basis.push(normalizeVector(vec));
        }
    }
    
    return basis;
}

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
    // Treat very small values as zero
    if (Math.abs(num) < 1e-9) {
        return '0';
    }
    
    // Round to 4 decimal places to avoid floating point noise
    const rounded = Math.round(num * 1e4) / 1e4;
    
    // If it's effectively an integer, show it as one
    if (Math.abs(rounded - Math.round(rounded)) < 1e-9) {
        return Math.round(rounded).toString();
    }
    
    // Check for common fractions
    const fracs = [
        { num: 1, den: 2 }, { num: 1, den: 3 }, { num: 2, den: 3 },
        { num: 1, den: 4 }, { num: 3, den: 4 }, { num: 1, den: 5 },
        { num: 2, den: 5 }, { num: 3, den: 5 }, { num: 4, den: 5 }
    ];
    
    for (const f of fracs) {
        if (Math.abs(Math.abs(rounded) - f.num / f.den) < 0.001) {
            const sign = rounded < 0 ? '-' : '';
            return `${sign}${f.num}/${f.den}`;
        }
    }
    
    // Otherwise format with up to 4 decimal places
    return parseFloat(rounded.toPrecision(4)).toString();
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
 * Primary: Uses math.eigs() which returns values and vectors.
 * Fallback: For defective matrices, computes eigenvalues separately then
 *           finds eigenvectors by solving the null space of (A - λI).
 * 
 * @param {number[][]} matrix 
 * @returns {{ eigenData: Array, matrix: number[][] }}
 */
function computeEigenData(matrix) {
    let result;
    try {
        // Try the standard math.js approach first
        result = computeEigenDataWithMathJS(matrix);
    } catch (err) {
        // If math.js fails (e.g., defective matrix), use fallback
        console.warn('math.js failed, using fallback:', err.message);
        result = computeEigenDataFallback(matrix);
    }
    
    // Sort eigenvalues by standard convention:
    // 1. Descending Magnitude (|z|)
    // 2. Descending Real Part
    // 3. Descending Imaginary Part
    result.eigenData.sort((a, b) => {
        const valA = a.eigenvalue;
        const valB = b.eigenvalue;
        
        const magA = complexMagnitude(valA);
        const magB = complexMagnitude(valB);
        
        // 1. Magnitude Descending
        if (Math.abs(magA - magB) > 1e-9) {
            return magB - magA;
        }
        
        // 2. Real Part Descending
        const reA = typeof valA === 'number' ? valA : valA.re;
        const reB = typeof valB === 'number' ? valB : valB.re;
        if (Math.abs(reA - reB) > 1e-9) {
            return reB - reA;
        }
        
        // 3. Imaginary Part Descending
        const imA = typeof valA === 'number' ? 0 : valA.im;
        const imB = typeof valB === 'number' ? 0 : valB.im;
        return imB - imA;
    });
    
    return result;
}

/**
 * Standard computation using math.js eigs().
 */
function computeEigenDataWithMathJS(matrix) {
    const result = math.eigs(matrix);
    
    // Get values as a plain array
    const valuesArray = result.values.toArray ? result.values.toArray() : 
                        (Array.isArray(result.values) ? result.values : result.values.valueOf());
    
    // Get vectors matrix and extract columns
    const vectorsMatrix = result.vectors.toArray ? result.vectors.toArray() : 
                          (Array.isArray(result.vectors) ? result.vectors : result.vectors.valueOf());
    
    const n = valuesArray.length;
    
    // Extract column vectors from the matrix
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
    
    // Filter each group's eigenvectors to only keep linearly independent ones
    const eigenDataArray = Array.from(eigenMap.values()).map(data => ({
        eigenvalue: data.eigenvalue,
        eigenvectors: filterLinearlyIndependent(data.eigenvectors),
        algebraicMultiplicity: data.algebraicMultiplicity
    }));
    
    return { eigenData: eigenDataArray, matrix };
}

/**
 * Fallback for defective matrices.
 * Computes eigenvalues using characteristic polynomial roots,
 * then finds eigenvectors by computing null space of (A - λI).
 */
/**
 * Fallback for defective/singular matrices where math.eigs fails.
 * Uses a robust 3-step process:
 * 1. Compute characteristic polynomial coefficients (Newton Sums)
 * 2. Find eigenvalues (roots) using Durand-Kerner method
 * 3. Find eigenvectors using Null Space of (A - λI)
 */
function computeEigenDataFallback(matrix) {
    const n = matrix.length;
    console.log("Using robust fallback solver...");

    // 1. Get coefficients of characteristic polynomial
    const coeffs = computeCharacteristicPolyCoeffs(matrix);
    
    // 2. Find roots (eigenvalues)
    const roots = durandKerner(coeffs);
    
    // Clean roots (snap tiny imaginary parts to 0) to ensure correct grouping
    const cleanedRoots = roots.map(r => cleanComplex(r));
    
    // 3. Group eigenvalues (fuzzy matching)
    const eigenMap = new Map();
    
    cleanedRoots.forEach(root => {
        let found = false;
        
        for (const [key, data] of eigenMap) {
            if (complexApproxEqual(key, root)) {
                data.algebraicMultiplicity++;
                found = true;
                break;
            }
        }
        
        if (!found) {
            eigenMap.set(root, {
                eigenvalue: root,
                algebraicMultiplicity: 1,
                eigenvectors: []
            });
        }
    });

    // 4. Compute eigenvectors for each unique eigenvalue
    const eigenDataArray = [];
    
    for (const data of eigenMap.values()) {
        const lambda = data.eigenvalue;
        
        // Use the complex null space solver
        const vectors = computeComplexNullSpace(matrix, lambda);
        
        // Normalize
        const normalizedVectors = vectors.map(v => normalizeComplexVector(v));
        
        eigenDataArray.push({
            eigenvalue: lambda,
            eigenvectors: normalizedVectors,
            algebraicMultiplicity: data.algebraicMultiplicity
        });
    }
    
    // Sort logic
    // Sorting is handled by the caller (computeEigenData)
    return { eigenData: eigenDataArray, matrix };
}

/**
 * Snaps tiny components of complex number to 0.
 */
function cleanComplex(c, tol = 1e-6) {
    let re = typeof c === 'number' ? c : c.re;
    let im = typeof c === 'number' ? 0 : c.im;
    
    if (Math.abs(re) < tol) re = 0;
    if (Math.abs(im) < tol) im = 0;
    
    // Also snap integers if very close (e.g. 4.9999999 -> 5)
    if (Math.abs(re - Math.round(re)) < tol) re = Math.round(re);
    if (Math.abs(im - Math.round(im)) < tol) im = Math.round(im);
    
    return math.complex(re, im);
}

/**
 * Computes char poly coeffs using Newton Sums.
 * Returns [c_0, c_1, ..., c_n] where P(x) = sum(c_i * x^i)
 */
function computeCharacteristicPolyCoeffs(matrix) {
    const n = matrix.length;
    const traces = [];
    let currentPower = matrix;
    
    for (let k = 1; k <= n; k++) {
        let tr = math.complex(0, 0);
        for (let i = 0; i < n; i++) {
             let diag = currentPower[i][i];
             // Ensure complex type
             if (typeof diag === 'number') diag = math.complex(diag, 0);
             tr = math.add(tr, diag);
        }
        traces.push(tr);
        
        if (k < n) {
            currentPower = math.multiply(currentPower, matrix);
        }
    }
    
    const coeffs = new Array(n + 1).fill(math.complex(0,0));
    coeffs[n] = math.complex(1, 0);
    
    for (let k = 1; k <= n; k++) {
        let sum = math.complex(0, 0);
        for (let j = 1; j <= k; j++) {
            const term = math.multiply(coeffs[n - k + j], traces[j-1]); 
            sum = math.add(sum, term);
        }
        coeffs[n - k] = math.divide(math.unaryMinus(sum), k);
    }
    
    return coeffs;
}

/**
 * Finds roots using Durand-Kerner method.
 */
function durandKerner(coeffs) {
    const n = coeffs.length - 1;
    let roots = [];
    
    // Radius R = 1 + max(|c_i|)
    let maxCoeffMag = 0;
    for (let i = 0; i < n; i++) {
        const mag = complexMagnitude(coeffs[i]);
        if (mag > maxCoeffMag) maxCoeffMag = mag;
    }
    const R = 1 + maxCoeffMag;
    
    for (let k = 0; k < n; k++) {
        const theta = (2 * Math.PI * k) / n + 0.1; 
        roots.push(math.multiply(R, math.complex({r: 1, phi: theta})));
    }
    
    for (let iter = 0; iter < 50; iter++) {
        let maxChange = 0;
        const nextRoots = [...roots];
        
        for (let i = 0; i < n; i++) {
            const z = roots[i];
            let pVal = math.complex(0, 0);
            for (let j = n; j >= 0; j--) {
                pVal = math.add(math.multiply(pVal, z), coeffs[j]);
            }
            
            let qVal = math.complex(1, 0);
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    qVal = math.multiply(qVal, math.subtract(z, roots[j]));
                }
            }
            
            const delta = math.divide(pVal, qVal);
            nextRoots[i] = math.subtract(z, delta);
            
            const mag = complexMagnitude(delta);
            if (mag > maxChange) maxChange = mag;
        }
        roots = nextRoots;
        if (maxChange < 1e-12) break;
    }
    return roots;
}

/**
 * Computes eigenvalues by finding roots of characteristic polynomial.
 * Uses math.js for polynomial root finding.
 */
function computeEigenvalues(matrix) {
    const n = matrix.length;
    
    // For small matrices, we can compute eigenvalues analytically or via numeric methods
    // Use companion matrix approach or direct computation
    
    if (n === 2) {
        // 2x2: λ² - trace(A)λ + det(A) = 0
        const trace = matrix[0][0] + matrix[1][1];
        const det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        const discriminant = trace * trace - 4 * det;
        
        if (discriminant >= 0) {
            const sqrtD = Math.sqrt(discriminant);
            return [(trace + sqrtD) / 2, (trace - sqrtD) / 2];
        } else {
            const sqrtD = Math.sqrt(-discriminant);
            return [
                math.complex(trace / 2, sqrtD / 2),
                math.complex(trace / 2, -sqrtD / 2)
            ];
        }
    }
    
    // For larger matrices, use QR iteration or fallback to diagonal entries for triangular
    // Check if matrix is triangular
    if (isUpperTriangular(matrix) || isLowerTriangular(matrix)) {
        return matrix.map((row, i) => row[i]); // Diagonal entries
    }
    
    // General case: Use power iteration / QR algorithm
    // For simplicity, try to get eigenvalues from math.js eigs with precision option
    try {
        const result = math.eigs(matrix, { precision: 1e-12 });
        const vals = result.values.toArray ? result.values.toArray() : result.values;
        return vals;
    } catch (e) {
        // Last resort: return diagonal entries as approximation
        return matrix.map((row, i) => row[i]);
    }
}

/**
 * Checks if matrix is upper triangular.
 */
function isUpperTriangular(matrix) {
    const n = matrix.length;
    for (let i = 1; i < n; i++) {
        for (let j = 0; j < i; j++) {
            if (Math.abs(matrix[i][j]) > TOLERANCE) return false;
        }
    }
    return true;
}

/**
 * Checks if matrix is lower triangular.
 */
function isLowerTriangular(matrix) {
    const n = matrix.length;
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            if (Math.abs(matrix[i][j]) > TOLERANCE) return false;
        }
    }
    return true;
}

/**
 * Computes the null space of (A - λI) to find eigenvectors.
 * Uses Gaussian elimination to find the kernel.
 * Supports complex eigenvalues using math.js complex arithmetic.
 */
function computeNullSpace(matrix, eigenvalue) {
    const n = matrix.length;
    
    // Check if eigenvalue is complex
    const isComplexLambda = eigenvalue && typeof eigenvalue === 'object' && 'im' in eigenvalue && Math.abs(eigenvalue.im) > TOLERANCE;
    
    if (isComplexLambda) {
        // Use complex arithmetic for complex eigenvalues
        return computeComplexNullSpace(matrix, eigenvalue);
    }
    
    const lambda = typeof eigenvalue === 'number' ? eigenvalue : eigenvalue.re;
    
    // Create (A - λI)
    const AminusLambdaI = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
            const val = typeof matrix[i][j] === 'number' ? matrix[i][j] : matrix[i][j].re;
            row.push(i === j ? val - lambda : val);
        }
        AminusLambdaI.push(row);
    }
    
    // Perform Gaussian elimination to find RREF
    const rref = gaussianElimination(AminusLambdaI);
    
    // Find free variables and construct basis vectors
    const pivotCols = [];
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (Math.abs(rref[i][j]) > TOLERANCE) {
                pivotCols.push(j);
                break;
            }
        }
    }
    
    const freeCols = [];
    for (let j = 0; j < n; j++) {
        if (!pivotCols.includes(j)) {
            freeCols.push(j);
        }
    }
    
    // If no free variables, matrix might be full rank (shouldn't happen for eigenvalue)
    if (freeCols.length === 0) {
        // Return a default vector
        const vec = new Array(n).fill(0);
        vec[0] = 1;
        return [vec];
    }
    
    // Construct basis vectors for null space
    const nullSpaceBasis = [];
    
    for (const freeCol of freeCols) {
        const vec = new Array(n).fill(0);
        vec[freeCol] = 1;
        
        // Back substitute to find other components
        for (let i = 0; i < pivotCols.length; i++) {
            const pivotCol = pivotCols[i];
            if (pivotCol < n) {
                vec[pivotCol] = -rref[i][freeCol];
            }
        }
        
        nullSpaceBasis.push(normalizeVector(vec));
    }
    
    return nullSpaceBasis;
}

/**
 * Computes the null space of (A - λI) for complex eigenvalues.
 * Uses math.js complex arithmetic throughout.
 */
function computeComplexNullSpace(matrix, eigenvalue) {
    const n = matrix.length;
    
    // Create (A - λI) with complex numbers
    const AminusLambdaI = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
            const matVal = typeof matrix[i][j] === 'number' ? 
                math.complex(matrix[i][j], 0) : matrix[i][j];
            if (i === j) {
                row.push(math.subtract(matVal, eigenvalue));
            } else {
                row.push(matVal);
            }
        }
        AminusLambdaI.push(row);
    }
    
    // Perform complex Gaussian elimination
    const rref = complexGaussianElimination(AminusLambdaI);
    
    // Find pivot columns
    const pivotCols = [];
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const val = rref[i][j];
            const mag = typeof val === 'number' ? Math.abs(val) : 
                Math.sqrt(val.re * val.re + val.im * val.im);
            if (mag > TOLERANCE) {
                pivotCols.push(j);
                break;
            }
        }
    }
    
    // Find free columns
    const freeCols = [];
    for (let j = 0; j < n; j++) {
        if (!pivotCols.includes(j)) {
            freeCols.push(j);
        }
    }
    
    // If no free variables, return default
    if (freeCols.length === 0) {
        const vec = [];
        for (let i = 0; i < n; i++) {
            vec.push(i === 0 ? math.complex(1, 0) : math.complex(0, 0));
        }
        return [vec];
    }
    
    // Construct basis vectors
    const nullSpaceBasis = [];
    for (const freeCol of freeCols) {
        const vec = [];
        for (let i = 0; i < n; i++) {
            vec.push(math.complex(0, 0));
        }
        vec[freeCol] = math.complex(1, 0);
        
        // Back substitute
        for (let i = 0; i < pivotCols.length; i++) {
            const pivotCol = pivotCols[i];
            if (pivotCol < n) {
                vec[pivotCol] = math.unaryMinus(rref[i][freeCol]);
            }
        }
        
        nullSpaceBasis.push(normalizeComplexVector(vec));
    }
    
    return nullSpaceBasis;
}

/**
 * Performs Gaussian elimination with complex numbers.
 */
function complexGaussianElimination(matrix) {
    const n = matrix.length;
    const m = matrix[0].length;
    
    // Create a deep copy
    const A = matrix.map(row => row.map(x => 
        typeof x === 'number' ? math.complex(x, 0) : math.complex(x.re, x.im)
    ));
    
    let pivotRow = 0;
    
    for (let col = 0; col < m && pivotRow < n; col++) {
        // Find pivot (row with largest magnitude)
        let maxRow = pivotRow;
        let maxMag = complexMagnitude(A[pivotRow][col]);
        
        for (let row = pivotRow + 1; row < n; row++) {
            const mag = complexMagnitude(A[row][col]);
            if (mag > maxMag) {
                maxMag = mag;
                maxRow = row;
            }
        }
        
        if (maxMag < TOLERANCE) {
            continue; // No pivot in this column
        }
        
        // Swap rows
        [A[pivotRow], A[maxRow]] = [A[maxRow], A[pivotRow]];
        
        // Scale pivot row
        const pivot = A[pivotRow][col];
        for (let j = col; j < m; j++) {
            A[pivotRow][j] = math.divide(A[pivotRow][j], pivot);
        }
        
        // Eliminate column
        for (let row = 0; row < n; row++) {
            if (row !== pivotRow && complexMagnitude(A[row][col]) > TOLERANCE) {
                const factor = A[row][col];
                for (let j = col; j < m; j++) {
                    A[row][j] = math.subtract(A[row][j], math.multiply(factor, A[pivotRow][j]));
                }
            }
        }
        
        pivotRow++;
    }
    
    return A;
}

/**
 * Returns magnitude of a complex number.
 */
function complexMagnitude(c) {
    if (typeof c === 'number') return Math.abs(c);
    return Math.sqrt(c.re * c.re + c.im * c.im);
}

/**
 * Normalizes a complex vector.
 */
function normalizeComplexVector(vec) {
    let sumSq = 0;
    for (const c of vec) {
        const re = typeof c === 'number' ? c : c.re;
        const im = typeof c === 'number' ? 0 : c.im;
        sumSq += re * re + im * im;
    }
    const norm = Math.sqrt(sumSq);
    if (norm < TOLERANCE) return vec;
    
    return vec.map(c => {
        if (typeof c === 'number') {
            return c / norm;
        }
        return math.complex(c.re / norm, c.im / norm);
    });
}

/**
 * Performs Gaussian elimination to get row echelon form.
 */
function gaussianElimination(matrix) {
    const n = matrix.length;
    const m = matrix[0].length;
    
    // Create a copy
    const A = matrix.map(row => [...row]);
    
    let pivotRow = 0;
    
    for (let col = 0; col < m && pivotRow < n; col++) {
        // Find pivot
        let maxRow = pivotRow;
        for (let row = pivotRow + 1; row < n; row++) {
            if (Math.abs(A[row][col]) > Math.abs(A[maxRow][col])) {
                maxRow = row;
            }
        }
        
        if (Math.abs(A[maxRow][col]) < TOLERANCE) {
            continue; // No pivot in this column
        }
        
        // Swap rows
        [A[pivotRow], A[maxRow]] = [A[maxRow], A[pivotRow]];
        
        // Scale pivot row
        const pivot = A[pivotRow][col];
        for (let j = col; j < m; j++) {
            A[pivotRow][j] /= pivot;
        }
        
        // Eliminate column
        for (let row = 0; row < n; row++) {
            if (row !== pivotRow && Math.abs(A[row][col]) > TOLERANCE) {
                const factor = A[row][col];
                for (let j = col; j < m; j++) {
                    A[row][j] -= factor * A[pivotRow][j];
                }
            }
        }
        
        pivotRow++;
    }
    
    return A;
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
    
    // Get vector array and format based on setting
    let vecArray = vector.toArray ? vector.toArray() : [...vector];
    let displayValues;
    
    if (vectorFormat === 'simplified') {
        // Integer format with canonical sign
        const simplified = simplifyVector(vecArray);
        displayValues = simplified.map(v => formatComplex(v));
    } else if (vectorFormat === 'fractions') {
        // Fraction format: simplify first, then convert to fractions
        const simplified = simplifyVector(vecArray);
        displayValues = vectorToFractions(simplified);
    } else {
        // Normalized: unit vector (re-normalize to ensure magnitude = 1)
        const normalized = normalizeVector(vecArray);
        displayValues = normalized.map(v => formatComplex(v));
    }
    
    displayValues.forEach(val => {
        const compEl = document.createElement('div');
        compEl.className = 'eigenvector__component';
        compEl.textContent = val;
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
 * @param {number[][]} matrix - The original matrix for verification.
 * @param {number} index - For staggered animation delay.
 * @returns {HTMLElement}
 */
function renderResultCard(data, matrix, index) {
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
 * @param {number[][]} matrix - The original matrix for verification.
 */
function renderResults(eigenData, matrix) {
    // Store for re-rendering when format changes
    lastEigenData = eigenData;
    lastMatrix = matrix;
    
    resultsSection.innerHTML = '';
    
    if (eigenData.length === 0) {
        showError('No eigenvalues found. The matrix may be singular or ill-conditioned.');
        resultsToolbar.setAttribute('aria-hidden', 'true');
        return;
    }
    
    // Show the results toolbar
    resultsToolbar.setAttribute('aria-hidden', 'false');
    
    eigenData.forEach((data, index) => {
        resultsSection.appendChild(renderResultCard(data, matrix, index));
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

/**
 * Handles dimension button click.
 * @param {Event} event - The click event.
 */
function handleDimensionChange(event) {
    const btn = event.target.closest('.dimension-btn');
    if (!btn) return;
    
    const newN = parseInt(btn.dataset.dimension, 10);
    if (newN === currentN) return;
    
    // Update active state
    dimensionBtns.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-checked', 'false');
    });
    btn.classList.add('active');
    btn.setAttribute('aria-checked', 'true');
    
    currentN = newN;
    generateGrid(currentN);
    resultsSection.innerHTML = '';
    resultsToolbar.setAttribute('aria-hidden', 'true');
    hideError();
}

function handleCompute() {
    hideError();
    resultsSection.innerHTML = '';
    
    try {
        const matrix = getMatrix();
        const { eigenData, matrix: originalMatrix } = computeEigenData(matrix);
        renderResults(eigenData, originalMatrix);
    } catch (err) {
        console.error('Computation error:', err);
        showError(`Computation failed: ${err.message || 'Unknown error'}`);
    }
}

// ==========================================================================
// Initialization
// ==========================================================================

/**
 * Test samples as specified in the verification requirements.
 */
const TEST_SAMPLES = [
    {
        name: 'Sample 1: 2×2 Distinct Real',
        matrix: [
            [4, 1],
            [2, 3]
        ],
        expectedEigenvalues: [2, 5]
    },
    {
        name: 'Sample 2: 3×3 Lower Triangular',
        matrix: [
            [2, 0, 0],
            [1, 4, 0],
            [-1, 3, 5]
        ],
        expectedEigenvalues: [2, 4, 5]
    },
    {
        name: 'Sample 3: 2×2 Defective (Repeated λ)',
        matrix: [
            [4, 1],
            [0, 4]
        ],
        expectedEigenvalues: [4, 4],
        expectedBasisCount: { 4: 1 } // Only 1 eigenvector for λ=4 (defective matrix)
    },
    {
        name: 'Sample 4: 3×3 Repeated λ, Full Basis',
        matrix: [
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 5]
        ],
        expectedEigenvalues: [3, 3, 5],
        expectedBasisCount: { 3: 2, 5: 1 } // 2 eigenvectors for λ=3
    },
    {
        name: 'Sample 5: 2×2 Rotation (Complex λ = ±i)',
        matrix: [
            [0, -1],
            [1, 0]
        ],
        expectedComplexEigenvalues: true, // Flag to indicate complex eigenvalue test
        expectedEigenvalueCount: 2
    },
    {
        name: 'Sample 6: 3×3 with Complex Eigenvalues',
        matrix: [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 2]
        ],
        expectedComplexEigenvalues: true,
        expectedEigenvalueCount: 3 // +i, -i, 2
    }
];

/**
 * Populates the matrix grid with the given matrix values.
 * @param {number[][]} matrix
 */
function populateMatrix(matrix) {
    const n = matrix.length;
    
    // Update dimension if needed
    if (n !== currentN) {
        currentN = n;
        
        // Update button states
        dimensionBtns.forEach(btn => {
            const dim = parseInt(btn.dataset.dimension, 10);
            if (dim === n) {
                btn.classList.add('active');
                btn.setAttribute('aria-checked', 'true');
            } else {
                btn.classList.remove('active');
                btn.setAttribute('aria-checked', 'false');
            }
        });
        
        generateGrid(n);
    }
    
    // Fill in values
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const input = document.getElementById(`cell-${i}-${j}`);
            input.value = matrix[i][j];
        }
    }
}

/**
 * Runs all test samples sequentially.
 */
function runTests() {
    hideError();
    resultsSection.innerHTML = '';
    
    // Create test results container
    const testContainer = document.createElement('div');
    testContainer.className = 'test-results';
    
    const header = document.createElement('h2');
    header.className = 'test-results__header';
    header.textContent = 'Verification Test Suite';
    testContainer.appendChild(header);
    
    let allPassed = true;
    
    for (const sample of TEST_SAMPLES) {
        const testResult = document.createElement('div');
        testResult.className = 'test-result';
        
        const testName = document.createElement('div');
        testName.className = 'test-result__name';
        testName.textContent = sample.name;
        testResult.appendChild(testName);
        
        try {
            populateMatrix(sample.matrix);
            const { eigenData, matrix } = computeEigenData(sample.matrix);
            
            let eigenMatch = true;
            let eigenDescription = '';
            
            // Handle complex eigenvalue tests differently
            if (sample.expectedComplexEigenvalues) {
                // For complex tests, verify count and residuals
                let totalEigenvalues = 0;
                eigenData.forEach(d => {
                    totalEigenvalues += d.algebraicMultiplicity;
                });
                eigenMatch = totalEigenvalues === sample.expectedEigenvalueCount;
                
                // Build description with complex values
                const eigenStrings = eigenData.map(d => {
                    const ev = d.eigenvalue;
                    if (typeof ev === 'object' && 'im' in ev && Math.abs(ev.im) > 1e-9) {
                        const sign = ev.im >= 0 ? '+' : '';
                        return `${ev.re.toFixed(4)}${sign}${ev.im.toFixed(4)}i`;
                    }
                    return (typeof ev === 'number' ? ev : ev.re).toFixed(4);
                });
                eigenDescription = `Count: ${totalEigenvalues}, Values: [${eigenStrings.join(', ')}]`;
            } else {
                // Original logic for real eigenvalue tests
                const foundEigenvalues = [];
                eigenData.forEach(d => {
                    for (let i = 0; i < d.algebraicMultiplicity; i++) {
                        const val = typeof d.eigenvalue === 'number' ? d.eigenvalue : d.eigenvalue.re;
                        foundEigenvalues.push(Math.round(val * 1e6) / 1e6);
                    }
                });
                foundEigenvalues.sort((a, b) => a - b);
                const expectedSorted = [...sample.expectedEigenvalues].sort((a, b) => a - b);
                eigenMatch = JSON.stringify(foundEigenvalues) === JSON.stringify(expectedSorted);
                eigenDescription = `Found: [${foundEigenvalues.join(', ')}]`;
            }
            
            // Check basis counts if specified
            let basisMatch = true;
            if (sample.expectedBasisCount) {
                for (const [eigenStr, expectedCount] of Object.entries(sample.expectedBasisCount)) {
                    const eigenVal = parseFloat(eigenStr);
                    const eigenGroup = eigenData.find(d => {
                        const val = typeof d.eigenvalue === 'number' ? d.eigenvalue : d.eigenvalue.re;
                        return Math.abs(val - eigenVal) < 1e-6;
                    });
                    if (!eigenGroup || eigenGroup.eigenvectors.length !== expectedCount) {
                        basisMatch = false;
                    }
                }
            }
            
            // Check residuals (this now works for complex eigenvalues too)
            let residualsPass = true;
            let maxResidual = 0;
            eigenData.forEach(d => {
                d.eigenvectors.forEach(vec => {
                    const residual = computeResidual(matrix, vec, d.eigenvalue);
                    maxResidual = Math.max(maxResidual, residual);
                    if (residual > 1e-6) residualsPass = false;
                });
            });
            
            const testPassed = eigenMatch && basisMatch && residualsPass;
            if (!testPassed) allPassed = false;
            
            const statusBadge = document.createElement('span');
            statusBadge.className = `test-result__status ${testPassed ? 'test-result__status--pass' : 'test-result__status--fail'}`;
            statusBadge.textContent = testPassed ? 'PASS' : 'FAIL';
            testResult.appendChild(statusBadge);
            
            // Details
            const details = document.createElement('div');
            details.className = 'test-result__details';
            details.innerHTML = `
                <div>Eigenvalues: ${eigenMatch ? '✓' : '✗'} (${eigenDescription})</div>
                <div>Basis Validity: ${basisMatch ? '✓' : '✗'}</div>
                <div>Residuals: ${residualsPass ? '✓' : '✗'} (max: ${maxResidual.toExponential(2)})</div>
            `;
            testResult.appendChild(details);
            
        } catch (err) {
            // Check if this is an expected library limitation
            if (sample.expectLibraryError && err.message.includes(sample.expectedError)) {
                // Mark as SKIP - known library limitation
                const statusBadge = document.createElement('span');
                statusBadge.className = 'test-result__status test-result__status--skip';
                statusBadge.textContent = 'SKIP';
                testResult.appendChild(statusBadge);
                
                const details = document.createElement('div');
                details.className = 'test-result__details';
                details.innerHTML = `<div>Known Limitation: math.js does not support defective matrices</div>`;
                testResult.appendChild(details);
            } else {
                allPassed = false;
                const statusBadge = document.createElement('span');
                statusBadge.className = 'test-result__status test-result__status--fail';
                statusBadge.textContent = 'ERROR';
                testResult.appendChild(statusBadge);
                
                const errDetails = document.createElement('div');
                errDetails.className = 'test-result__details';
                errDetails.textContent = err.message;
                testResult.appendChild(errDetails);
            }
        }
        
        testContainer.appendChild(testResult);
    }
    
    // Summary
    const summary = document.createElement('div');
    summary.className = `test-results__summary ${allPassed ? 'test-results__summary--pass' : 'test-results__summary--fail'}`;
    summary.textContent = allPassed ? '✓ All tests passed!' : '✗ Some tests failed';
    testContainer.appendChild(summary);
    
    resultsSection.appendChild(testContainer);
}

function init() {
    // Set up event listeners for dimension buttons
    dimensionSelector.addEventListener('click', handleDimensionChange);
    computeBtn.addEventListener('click', handleCompute);
    
    // Clear button
    if (clearBtn) {
        clearBtn.addEventListener('click', handleClear);
    }
    
    // Test suite button
    const runTestsBtn = document.getElementById('run-tests-btn');
    if (runTestsBtn) {
        runTestsBtn.addEventListener('click', runTests);
    }
    
    // Allow Enter key to trigger computation when in an input
    matrixGrid.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            handleCompute();
        }
    });
    
    // Event listener for vector format toggle
    formatSelector.addEventListener('click', handleFormatChange);
    
    // Initial grid generation
    generateGrid(currentN);
}

/**
 * Handles clicks on the vector format buttons.
 */
function handleFormatChange(e) {
    if (e.target.classList.contains('format-btn')) {
        const selectedFormat = e.target.dataset.format;
        
        if (selectedFormat !== vectorFormat) {
            vectorFormat = selectedFormat;
            
            // Update active state
            formatBtns.forEach(btn => {
                if (btn.dataset.format === selectedFormat) {
                    btn.classList.add('active');
                    btn.setAttribute('aria-checked', 'true');
                } else {
                    btn.classList.remove('active');
                    btn.setAttribute('aria-checked', 'false');
                }
            });
            
            // Re-render results if we have data
            if (lastEigenData && lastMatrix) {
                renderResults(lastEigenData, lastMatrix);
            }
        }
    }
}

// Run on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
/**
 * Handles clearing the matrix input and results.
 */
function handleClear() {
    // Reset all inputs to empty
    const inputs = matrixGrid.querySelectorAll('input');
    inputs.forEach(input => {
        input.value = '';
    });
    
    // Clear results
    resultsSection.innerHTML = '';
    resultsToolbar.setAttribute('aria-hidden', 'true');
    hideError();
    
    // Clear state
    lastEigenData = null;
    lastMatrix = null;
}
