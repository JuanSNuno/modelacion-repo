// ============================================================
// Lab Controller — ML Model Training & Comparison
// ============================================================

import { Request, Response } from 'express';
import * as modelService from '../services/model.service';

/**
 * GET /api/lab/models — Get catalog of available models
 */
export function getCatalog(_req: Request, res: Response) {
    const catalog = modelService.getModelCatalog();
    return res.json(catalog);
}

/**
 * POST /api/lab/train — Train a specific model
 */
export function train(req: Request, res: Response) {
    try {
        const { modelType, X, y, params } = req.body;

        if (!modelType || !X || !y) {
            return res.status(400).json({ error: 'Campos requeridos: modelType, X, y.' });
        }

        if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length) {
            return res.status(400).json({ error: 'X y y deben ser arrays de la misma longitud.' });
        }

        const result = modelService.trainModel({
            modelType,
            X,
            y,
            params: params || {}
        });

        return res.json(result);
    } catch (err: any) {
        return res.status(400).json({ error: err.message });
    }
}

/**
 * POST /api/lab/compare — Compare multiple models
 */
export function compare(req: Request, res: Response) {
    try {
        const { models, X, y } = req.body;

        if (!models || !Array.isArray(models) || !X || !y) {
            return res.status(400).json({ error: 'Campos requeridos: models (array), X, y.' });
        }

        const results = modelService.compareModels({ models, X, y });
        return res.json(results);
    } catch (err: any) {
        return res.status(400).json({ error: err.message });
    }
}

/**
 * POST /api/lab/save — Save a training result
 */
export async function saveResult(req: Request, res: Response) {
    try {
        const result = await modelService.saveTrainingResult(req.body);
        return res.status(201).json(result);
    } catch (err: any) {
        return res.status(500).json({ error: err.message });
    }
}

/**
 * GET /api/lab/history — Get training history
 */
export async function getHistory(_req: Request, res: Response) {
    try {
        const history = await modelService.getTrainingHistory();
        return res.json(history);
    } catch (err: any) {
        return res.status(500).json({ error: err.message });
    }
}
