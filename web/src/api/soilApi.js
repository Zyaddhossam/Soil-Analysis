const API_BASE = 'http://localhost:8000/api/v1/predictions';

export async function getModelInfo() {
  const response = await fetch(`${API_BASE}/model-info`);
  if (!response.ok) {
    throw new Error('Failed to fetch model info');
  }
  return response.json();
}

export async function classifySoilType(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(
    `${API_BASE}/soil-type?include_probabilities=true`,
    { method: 'POST', body: formData }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Soil type classification failed');
  }

  return response.json();
}

export async function predictFertility(nutrients) {
  const response = await fetch(
    `${API_BASE}/fertility?include_probabilities=true`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(nutrients),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Fertility prediction failed');
  }

  return response.json();
}

export async function analyzeSoil(imageFile, nutrients) {
  const formData = new FormData();
  formData.append('file', imageFile);

  for (const [key, value] of Object.entries(nutrients)) {
    formData.append(key, value);
  }

  const response = await fetch(
    `${API_BASE}/analyze?include_probabilities=true`,
    { method: 'POST', body: formData }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Soil analysis failed');
  }

  return response.json();
}
