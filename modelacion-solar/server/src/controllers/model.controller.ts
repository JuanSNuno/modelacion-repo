// ============================================================
// Model Controller — Polynomial Regression (stitch original)
// ============================================================

import { Request, Response } from 'express';
import { polynomialRegression } from '../services/math.service';

export function calculateModel(req: Request, res: Response) {
    try {
        const { data, degree, experimentType } = req.body;

        if (!data || !Array.isArray(data) || data.length < 2) {
            return res.status(400).json({ error: 'Se requieren al menos 2 puntos de datos.' });
        }

        if (!degree || degree < 1 || degree > 10) {
            return res.status(400).json({ error: 'El grado debe estar entre 1 y 10.' });
        }

        const computeDerivative = experimentType === 'SOMBRA';
        const result = polynomialRegression(data, degree, computeDerivative);

        return res.json(result);
    } catch (err: any) {
        return res.status(400).json({ error: err.message });
    }
}
