const upload = document.getElementById('upload');

var image;

async function loadModel() {
    model = undefined;
    model = await tf.loadLayersModel('tmp/model/model.json');
}

loadModel();

function preprocessImage(image) {
    const resizedImage = tf.image.resizeBilinear(image, [224, 224]);
    const normalizedImage = resizedImage.div(255.0);
    const batchedImage = normalizedImage.expandDims(0);
    return batchedImage;
  }

function handleImageUpload(event) {
    const image = document.getElementById('image');
    image.src = URL.createObjectURL(event.target.files[0]);
    image.style.visibility = "visible";
    const result = document.getElementById("result");
    result.innerHTML = "";
}

function handlePrediction() {
    const image = document.getElementById("image");
    const imageTensor = tf.browser.fromPixels(image);
    const preprocessedImage = preprocessImage(imageTensor);
    const prediction = model.predict(preprocessedImage);
    const disease = tf.argMax(prediction, 1).dataSync()[0];
    const result = document.getElementById("result");
    if (disease===0){
        result.innerHTML = "Normal";
    }
    else{
        result.innerHTML = "Detected pnuemonia";
    }
};
