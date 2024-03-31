async function predictImage() {
    const input = document.getElementById('imageInput');
    const preview = document.getElementById('preview');
    const predictionResult = document.getElementById('predictionResult');

    if (input.files.length === 0) {
        alert('Please select an image before predicting.');
        return;
    }

    const file = input.files[0];
    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        // Display the prediction result
        predictionResult.innerText = `Prediction: ${result.prediction}`;

    } catch (error) {
        console.error('Prediction failed:', error);
        alert('Prediction failed. Please try again.');
    }
}

function previewImage() {
    const input = document.getElementById('imageInput');
    const preview = document.getElementById('preview');

    const file = input.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            preview.src = e.target.result;
        };

        reader.readAsDataURL(file);
    }
}
