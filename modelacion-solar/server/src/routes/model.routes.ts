// ============================================================
// Routes — /api/model
// ============================================================

import { Router } from 'express';
import { calculateModel } from '../controllers/model.controller';

const router = Router();

router.post('/calculate', calculateModel);

export default router;
