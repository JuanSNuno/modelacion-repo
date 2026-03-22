// ============================================================
// Statistical Metrics — R², MSE, MAE, RMSE
// ============================================================

/**
 * Mean Squared Error
 */
export function mse(actual: number[], predicted: number[]): number {
    const n = actual.length;
    if (n === 0) return 0;
    return actual.reduce((sum, y, i) => sum + (y - predicted[i]) ** 2, 0) / n;
}

/**
 * Root Mean Squared Error
 */
export function rmse(actual: number[], predicted: number[]): number {
    return Math.sqrt(mse(actual, predicted));
}

/**
 * Mean Absolute Error
 */
export function mae(actual: number[], predicted: number[]): number {
    const n = actual.length;
    if (n === 0) return 0;
    return actual.reduce((sum, y, i) => sum + Math.abs(y - predicted[i]), 0) / n;
}

/**
 * Coefficient of Determination (R²)
 */
export function rSquared(actual: number[], predicted: number[]): number {
    const n = actual.length;
    if (n === 0) return 0;
    const mean = actual.reduce((s, v) => s + v, 0) / n;
    const ssRes = actual.reduce((sum, y, i) => sum + (y - predicted[i]) ** 2, 0);
    const ssTot = actual.reduce((sum, y) => sum + (y - mean) ** 2, 0);
    if (ssTot === 0) return 1; // perfect model on constant data
    return 1 - ssRes / ssTot;
}

/**
 * Evaluate a polynomial at a given x value.
 * coefficients[0] = a_0, coefficients[1] = a_1, etc.
 * y(x) = a_0 + a_1*x + a_2*x^2 + ...
 */
export function evaluatePolynomial(coefficients: number[], x: number): number {
    return coefficients.reduce((sum, coeff, power) => sum + coeff * Math.pow(x, power), 0);
}

/**
 * Derive polynomial coefficients.
 * If y(x) = a_0 + a_1*x + a_2*x^2 + a_3*x^3
 * then y'(x) = a_1 + 2*a_2*x + 3*a_3*x^2
 */
export function derivePolynomial(coefficients: number[]): number[] {
    if (coefficients.length <= 1) return [0];
    return coefficients.slice(1).map((coeff, i) => coeff * (i + 1));
}

/**
 * Format polynomial as a human-readable string.
 * e.g. "y = 1.2x² - 0.5x + 3.0"
 */
export function formatPolynomial(coefficients: number[], variable = 'x', funcName = 'y'): string {
    if (coefficients.length === 0) return `${funcName} = 0`;

    const superscripts: Record<number, string> = { 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹', 10: '¹⁰' };

    const terms: string[] = [];

    // Process from highest degree to lowest for conventional display
    for (let i = coefficients.length - 1; i >= 0; i--) {
        const coeff = coefficients[i];
        if (Math.abs(coeff) < 1e-10) continue;

        const absCoeff = Math.abs(coeff);
        const sign = coeff >= 0 ? '+' : '-';

        let term = '';
        if (i === 0) {
            term = absCoeff.toFixed(2);
        } else if (i === 1) {
            term = absCoeff === 1 ? variable : `${absCoeff.toFixed(2)}${variable}`;
        } else {
            const exp = superscripts[i] || `^${i}`;
            term = absCoeff === 1 ? `${variable}${exp}` : `${absCoeff.toFixed(2)}${variable}${exp}`;
        }

        if (terms.length === 0) {
            terms.push(coeff < 0 ? `-${term}` : term);
        } else {
            terms.push(`${sign} ${term}`);
        }
    }

    return `${funcName} = ${terms.join(' ') || '0'}`;
}
