// ============================================================
// Routes — /api/experiments
// ============================================================

import { Router } from 'express';
import * as experimentController from '../controllers/experiment.controller';

const router = Router();

router.get('/', experimentController.getAll);
router.get('/:id', experimentController.getById);
router.post('/', experimentController.create);
router.delete('/:id', experimentController.remove);

export default router;
