import { Router } from 'express';
import type { Router as RouterType } from 'express';

import { geocodingService } from './geocoding.service.js';

const router: RouterType = Router();

// POST /api/map/geocoding/reverse
// Body: { cameraId: string, lat: number, lng: number }
router.post('/reverse', async (req, res) => {
  try {
    const { cameraId, lat, lng } = req.body;

    if (!cameraId || typeof cameraId !== 'string' || typeof lat !== 'number' || typeof lng !== 'number') {
      res.status(400).json({ error: 'Invalid input. Need cameraId, lat, lng.' });
      return;
    }

    const location_details = await geocodingService.updateCameraLocation(cameraId, lat, lng);
    res.json({ success: true, location_details });
  } catch (error) {
    console.error('Reverse Geocode error:', error);
    res.status(500).json({ error: (error as Error).message });
  }
});

export default router;
