import { homographyService } from '../homography/homography.service.js';

interface NominatimResponse {
    address: {
        road?: string;
        suburb?: string;
        neighbourhood?: string;
        city_district?: string;
        city?: string;
        town?: string;
        village?: string;
        county?: string;
        state?: string;
        province?: string;
        country?: string;
        country_code?: string;
    };
    display_name: string;
}

export class GeocodingService {
    private readonly USER_AGENT = 'CityAsASensor/1.0';

    /**
     * Reverse geocodes a lat/lng to get address details.
     */
    async reverseGeocode(lat: number, lng: number): Promise<NominatimResponse | null> {
        try {
            const url = `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json`;
            const response = await fetch(url, {
                headers: {
                    'User-Agent': this.USER_AGENT
                }
            });

            if (!response.ok) {
                if (response.status === 429) {
                    console.warn(`[Geocoding] Rate limit exceeded (429). Please wait before retrying.`);
                    throw new Error('Nominatim API rate limit exceeded. Please try again later.');
                }
                if (response.status >= 500) {
                    console.warn(`[Geocoding] External API server error (${response.status}).`);
                    throw new Error(`Nominatim API unavailable (Status ${response.status}).`);
                }
                console.warn(`[Geocoding] API error: ${response.status} ${response.statusText}`);
                return null; // For 4xx errors other than 429, it might mean no data found or bad request
            }

            return await response.json() as NominatimResponse;
        } catch (error) {
            console.error('[Geocoding] Request failed:', error);
            // Re-throw if it's our specific error, otherwise return null or generic error
            if (error instanceof Error && (error.message.includes('rate limit') || error.message.includes('unavailable'))) {
                throw error;
            }
            return null;
        }
    }

    /**
     * Updates a camera's location details using reverse geocoding.
     */
    async updateCameraLocation(cameraId: string, lat: number, lng: number) {
        const data = await this.reverseGeocode(lat, lng);
        if (!data || !data.address) {
            throw new Error('Failed to fetch address data');
        }

        const addr = data.address;

        // Map OSM fields to our schema
        const district = addr.suburb || addr.neighbourhood || addr.city_district;
        const region = addr.state || addr.province || addr.city || addr.town || addr.county;
        const country = addr.country;

        const locationDetails = {
            district,
            region,
            country
        };

        // Update via homography service (which manages persistence)
        const camera = await homographyService.getCamera(cameraId);
        if (camera) {
            camera.location = { lat, lng }; // Ensure lat/lng is synced
            camera.location_details = locationDetails;
            await homographyService.saveCamera(camera);
        }

        return locationDetails;
    }
}

export const geocodingService = new GeocodingService();
