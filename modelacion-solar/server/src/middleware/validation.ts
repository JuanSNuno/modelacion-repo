// ============================================================
// Validation Middleware
// ============================================================

import { Request, Response, NextFunction } from 'express';

/**
 * Validates that the request body contains the required fields.
 */
export function validateBody(...fields: string[]) {
    return (req: Request, res: Response, next: NextFunction) => {
        const missing = fields.filter(f => req.body[f] === undefined || req.body[f] === null);
        if (missing.length > 0) {
            return res.status(400).json({
                error: `Campos requeridos faltantes: ${missing.join(', ')}`
            });
        }
        next();
    };
}
