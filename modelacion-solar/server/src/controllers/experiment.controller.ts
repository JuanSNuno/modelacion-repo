// ============================================================
// Experiment Controller — CRUD endpoints
// ============================================================

import { Request, Response } from 'express';
import * as experimentService from '../services/experiment.service';

export async function getAll(req: Request, res: Response) {
    try {
        const experiments = await experimentService.getAllExperiments();
        // Parse coefficients from JSON string
        const parsed = experiments.map(e => ({
            ...e,
            coefficients: JSON.parse(e.coefficients as string)
        }));
        return res.json(parsed);
    } catch (err: any) {
        return res.status(500).json({ error: err.message });
    }
}

export async function getById(req: Request, res: Response) {
    try {
        const experiment = await experimentService.getExperimentById(req.params.id as string);
        if (!experiment) {
            return res.status(404).json({ error: 'Experimento no encontrado.' });
        }
        return res.json({
            ...experiment,
            coefficients: JSON.parse(experiment.coefficients as string)
        });
    } catch (err: any) {
        return res.status(500).json({ error: err.message });
    }
}

export async function create(req: Request, res: Response) {
    try {
        const { name, type, degree, coefficients, r2, mse, dataPoints } = req.body;

        if (!name || !type || !degree || !coefficients || !dataPoints) {
            return res.status(400).json({ error: 'Faltan campos requeridos: name, type, degree, coefficients, dataPoints.' });
        }

        const experiment = await experimentService.createExperiment({
            name, type, degree, coefficients, r2, mse, dataPoints
        });

        return res.status(201).json({
            ...experiment,
            coefficients: JSON.parse(experiment.coefficients as string)
        });
    } catch (err: any) {
        return res.status(500).json({ error: err.message });
    }
}

export async function remove(req: Request, res: Response) {
    try {
        await experimentService.deleteExperiment(req.params.id as string);
        return res.json({ message: 'Experimento eliminado correctamente.' });
    } catch (err: any) {
        return res.status(500).json({ error: err.message });
    }
}
