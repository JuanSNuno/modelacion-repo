// ============================================================
// Linear Regression — Ordinary Least Squares
// ============================================================

import { TrainableModel, ModelState, HyperparameterDef } from './base.model';
import { gaussElimination } from '../utils/matrix';

export class LinearRegressionModel implements TrainableModel {
    readonly name = 'LINEAR';
    readonly category = 'regression';
    readonly description = 'Regresión Lineal por Mínimos Cuadrados Ordinarios (OLS)';
    readonly hyperparameterSchema: HyperparameterDef[] = [];

    train(X: number[][], y: number[], _params: Record<string, number>): ModelState {
        const n = X.length;
        const features = X[0]?.length || 1;

        // Add bias column (intercept) — prepend 1 to each row
        const Xb = X.map(row => [1, ...row]);
        const m = features + 1;

        // Normal equations: (X^T * X) * beta = X^T * y
        const XtX: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
        const XtY: number[] = new Array(m).fill(0);

        for (let i = 0; i < n; i++) {
            for (let j = 0; j < m; j++) {
                XtY[j] += Xb[i][j] * y[i];
                for (let k = 0; k < m; k++) {
                    XtX[j][k] += Xb[i][j] * Xb[i][k];
                }
            }
        }

        const coefficients = gaussElimination(XtX, XtY);
        if (!coefficients) {
            throw new Error('Matriz singular en regresión lineal. Verifique los datos.');
        }

        return {
            modelType: this.name,
            coefficients,
            params: _params
        };
    }

    predict(state: ModelState, X: number[][]): number[] {
        const coeffs = state.coefficients!;
        return X.map(row => {
            let sum = coeffs[0]; // intercept
            for (let j = 0; j < row.length; j++) {
                sum += coeffs[j + 1] * row[j];
            }
            return sum;
        });
    }
}
