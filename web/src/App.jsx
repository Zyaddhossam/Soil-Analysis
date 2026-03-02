import { useAnalysis } from './hooks/useAnalysis';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import NutrientForm from './components/NutrientForm';
import AnalyzeButton from './components/AnalyzeButton';
import ResultsPanel from './components/ResultsPanel';
import ErrorAlert from './components/ErrorAlert';

function App() {
  const {
    image, setImage,
    nutrients, setNutrients,
    results, loading, error,
    hasImage, hasNutrients,
    analyze, reset, setError,
  } = useAnalysis();

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />

      <main className="max-w-6xl mx-auto px-4 py-8">
        <ErrorAlert message={error} onDismiss={() => setError(null)} />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-4">
          <div className="space-y-4">
            <ImageUploader image={image} onImageChange={setImage} />
            <NutrientForm
              nutrients={nutrients}
              onNutrientsChange={setNutrients}
              disabled={loading}
            />
            <AnalyzeButton
              hasImage={hasImage}
              hasNutrients={hasNutrients}
              loading={loading}
              onAnalyze={analyze}
            />
            {(results || hasImage || hasNutrients) && (
              <button
                onClick={reset}
                disabled={loading}
                className="w-full py-2 text-sm text-gray-500 hover:text-gray-700 disabled:opacity-50"
              >
                Reset All
              </button>
            )}
          </div>

          <div>
            <ResultsPanel results={results} loading={loading} />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
