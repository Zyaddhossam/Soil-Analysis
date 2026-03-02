import { useState, useCallback } from 'react';
import { classifySoilType, predictFertility, analyzeSoil } from '../api/soilApi';
import { EMPTY_NUTRIENTS } from '../constants/nutrients';

export function useAnalysis() {
  const [image, setImage] = useState(null);
  const [nutrients, setNutrients] = useState({ ...EMPTY_NUTRIENTS });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const hasImage = image !== null;
  const hasNutrients = Object.values(nutrients).some(v => v !== '' && v !== null);

  const analyze = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const parsedNutrients = {};
      for (const [key, value] of Object.entries(nutrients)) {
        if (value !== '' && value !== null) {
          parsedNutrients[key] = parseFloat(value);
        }
      }
      const allNutrientsFilled = Object.keys(parsedNutrients).length === 12;

      if (hasImage && allNutrientsFilled) {
        const data = await analyzeSoil(image, parsedNutrients);
        setResults({ soilType: data.soil_type, fertility: data.fertility });
      } else if (hasImage) {
        const data = await classifySoilType(image);
        setResults({ soilType: data, fertility: null });
      } else if (allNutrientsFilled) {
        const data = await predictFertility(parsedNutrients);
        setResults({ soilType: null, fertility: data });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [image, nutrients, hasImage]);

  const reset = useCallback(() => {
    setImage(null);
    setNutrients({ ...EMPTY_NUTRIENTS });
    setResults(null);
    setError(null);
  }, []);

  return {
    image, setImage,
    nutrients, setNutrients,
    results, loading, error,
    hasImage, hasNutrients,
    analyze, reset, setError,
  };
}
