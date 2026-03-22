// ============================================================
// Routes — /api/lab
// ============================================================

import { Router } from 'express';
import * as labController from '../controllers/lab.controller';

const router = Router();

router.get('/models', labController.getCatalog);
router.post('/train', labController.train);
router.post('/compare', labController.compare);
router.post('/save', labController.saveResult);
router.get('/history', labController.getHistory);

export default router;
