import { describe, it, expect, vi } from 'vitest';
import { writeMap } from './storage.js';

// Mock fs/promises
vi.mock('fs/promises', () => ({
    default: {
        mkdir: vi.fn().mockResolvedValue(undefined),
        writeFile: vi.fn().mockResolvedValue(undefined),
        readFile: vi.fn().mockResolvedValue('{}'),
        access: vi.fn().mockResolvedValue(undefined),
        unlink: vi.fn().mockResolvedValue(undefined),
    }
}));

describe('Storage Security', () => {
    it('should prevent path traversal in writeMap', async () => {
        const maliciousLocation = {
            country: '..',
            province: '..',
            district: 'etc',
            relationId: '123'
        };

        const maliciousGraph = { nodes: [], ways: [] };

        // Attempt to write using malicious path components
        // Expect it to fail with a security error
        await expect(writeMap(maliciousLocation, maliciousGraph))
            .rejects
            .toThrow('Security Error: Path traversal detected');
    });

    it('should allow valid paths in writeMap', async () => {
        const validLocation = {
            country: 'Country',
            province: 'Province',
            district: 'District',
            relationId: '123'
        };

        const validGraph = { nodes: [], ways: [] };

        // Should not throw
        await expect(writeMap(validLocation, validGraph)).resolves.not.toThrow();
    });
});
