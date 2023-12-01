import './App.css';
import React, { useState } from 'react';
import axios from 'axios';

const OCRApp = () => {
  const [transcribedText, setTranscribedText] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);

  // Function to handle image selection
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    // You can perform additional checks here, e.g., file type validation
    setSelectedImage(URL.createObjectURL(file));
    // Here, you can integrate your OCR functionality to transcribe text from the image
    // Once you have the transcribed text, update the state using setTranscribedText
    // setTranscribedText(transcribedTextFromOCR);
  };

  // Function to handle form submission
  const handleSubmit = () => {
    // Check if an image is selected
    if (selectedImage) {
      // Create a FormData object to send the image file
      const formData = new FormData();
      formData.append('image', selectedImage);

      // Make a POST request to the Flask backend
      axios.post('http://localhost:5000/predict', formData) // Replace with your backend URL
        .then(response => {
          // Assuming the response contains the transcribed text
          setTranscribedText(response.data.prediction); // Set the transcribed text in the state
        })
        .catch(error => {
          console.error('Error:', error);
        });
    } else {
      console.error('No image selected');
    }
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="bg-gray-800 text-white p-4">
        <h1 className="text-2xl font-bold">GatorNum</h1>
      </header>

      {/* Main Content */}
      <main
        className="flex-1 p-4 bg-cover bg-opacity bg-gradient-to-t from-gray-700 to-gray-900"
        style={{
          /* Credit: https://www.pexels.com/photo/blue-and-purple-cosmic-sky-956999/  */
          // backgroundImage: `url('/images/bg_image.jpg')`, // Add your image path here
        }}
      >
        <div className="mb-4">
          <h2 className="text-xl text-center font-bold mb-2 text-white">Description</h2>
          <p className="text-white text-center">Your description goes here...</p>
        </div>

        {/* Image Upload */}
        <div className="flex flex-col items-center justify-center">
          <div className="bg-white p-4 rounded-md">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="border border-gray-300 p-2 rounded-md"
            />
          </div>
        </div>
        {/* Image Placeholder */}
        {selectedImage && (
            <div className="mb-4 flex justify-center" style={{ paddingTop: '40px' }}>
            <img
              src={selectedImage}
              alt="Selected"
              className="max-w-3/4 h-auto rounded-md shadow-md"
            />
          </div>
        )}

        {/* Transcribed Text Box */}
        <div style={{ maxWidth: '75%', margin: '0 auto' }}>
          <h2 className="text-xl font-bold mb-2 text-center text-white p-2">Transcribed Text</h2>
          <textarea
            value={transcribedText}
            onChange={(e) => setTranscribedText(e.target.value)}
            className="border border-gray-300 p-2 rounded-md w-full h-32 resize-none"
            placeholder="Transcribed text will appear here..."
            readOnly
          ></textarea>
        </div>

        {/* Submit Button */}
        <button
          onClick={handleSubmit}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mx-auto mt-4"
          style={{ display: 'block' }}
        >
          Submit
        </button>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white p-4">
        <p>&copy; 2023 Gabriel Castejon & Larry Mason</p>
      </footer>
    </div>
  );
};

export default OCRApp;
        
