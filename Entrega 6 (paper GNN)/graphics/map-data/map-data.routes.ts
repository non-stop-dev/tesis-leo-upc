/**
 * @fileoverview Express routes for map data API.
 * @module map-data/map-data.routes
 */

import { Router, Request, Response } from 'express';
import type { Router as RouterType } from 'express';
import { 
    getDistrict, 
    refreshDistrict, 
    removeDistrict, 
    getAllDistricts 
} from './map-data.service.js';
import type { DistrictLocation } from './types.js';

const router: RouterType = Router();

/**
 * GET /api/map/districts
 * Lists all saved districts as a tree structure.
 */
router.get('/', async (_req: Request, res: Response) => {
    try {
        const districts = await getAllDistricts();
        res.json({ countries: districts });
    } catch (error) {
        console.error('[MapData] Error listing districts:', error);
        res.status(500).json({ 
            error: 'Failed to list districts',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});

/**
 * GET /api/map/districts/:country/:province/:district
 * Gets district data.
 * If cached, relationId is optional.
 * If not cached, relationId is required to fetch from Overpass.
 */
router.get('/:country/:province/:district', async (req: Request, res: Response) => {
    try {
        const { country, province, district } = req.params;
        const relationId = req.query.relationId as string || '';
        
        const location: DistrictLocation = { country, province, district, relationId };
        
        // Try to get district
        // If relationId is missing and not in cache, getDistrict will throw/fail
        try {
            const graph = await getDistrict(location);
            res.json(graph);
        } catch (err: any) {
            if (err.message?.includes('relationId required')) {
                res.status(400).json({ 
                    error: 'Missing required query parameter: relationId',
                    message: 'Relation ID is required to fetch new data from Overpass',
                    example: '/api/map/districts/peru/lima/santiago-de-surco?relationId=1944844'
                });
            } else {
                throw err;
            }
        }
    } catch (error) {
        console.error('[MapData] Error getting district:', error);
        res.status(500).json({ 
            error: 'Failed to get district',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});

/**
 * POST /api/map/districts/:country/:province/:district/refresh
 * Forces a re-fetch from Overpass API.
 */
router.post('/:country/:province/:district/refresh', async (req: Request, res: Response) => {
    try {
        const { country, province, district } = req.params;
        const relationId = req.query.relationId as string;
        
        if (!relationId) {
            res.status(400).json({ 
                error: 'Missing required query parameter: relationId'
            });
            return;
        }
        
        const location: DistrictLocation = { country, province, district, relationId };
        const graph = await refreshDistrict(location);
        
        res.json(graph);
    } catch (error) {
        console.error('[MapData] Error refreshing district:', error);
        res.status(500).json({ 
            error: 'Failed to refresh district',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});

/**
 * DELETE /api/map/districts/:country/:province/:district
 * Deletes a cached district.
 */
router.delete('/:country/:province/:district', async (req: Request, res: Response) => {
    try {
        const { country, province, district } = req.params;
        const relationId = req.query.relationId as string || '';
        
        const location: DistrictLocation = { country, province, district, relationId };
        const deleted = await removeDistrict(location);
        
        if (deleted) {
            res.json({ success: true, message: 'District deleted' });
        } else {
            res.status(404).json({ error: 'District not found' });
        }
    } catch (error) {
        console.error('[MapData] Error deleting district:', error);
        res.status(500).json({ 
            error: 'Failed to delete district',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});

export default router;
